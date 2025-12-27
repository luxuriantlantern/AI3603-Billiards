import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal

from bayes_opt import BayesianOptimization
from .agent import Agent

# ============ Helper Functions ============

class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟"""
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间

    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        # print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    """
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]

    # 根据 player_targets 判断进球归属
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]

    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break

    # 首球犯规判定
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True

    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False

    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True

    # 计算奖励分数
    score = 0

    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 150 if is_targeting_eight_ball_legally else -500

    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30

    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20

    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10

    return score

class NewAgent(Agent):
    """
    NewAgent: 结合启发式搜索与贝叶斯优化的混合 Agent
    1. 使用几何启发式方法生成候选击球动作。
    2. 筛选出最佳候选动作。
    3. 使用贝叶斯优化在最佳候选动作附近进行微调，以应对物理误差并寻找最优解。
    """

    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        # 贝叶斯优化参数
        self.bo_init_points = 2
        self.bo_n_iter = 5

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        actions = []
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids: target_ids = ['8']

        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 基础力度
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 生成几个变种
                actions.append({'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})
                actions.append({'V0': min(v_base + 1.0, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0})

        if not actions:
            for _ in range(5): actions.append(self._random_action())

        random.shuffle(actions)
        return actions[:20] # 限制数量

    def _simulate_and_score(self, action, balls, table, my_targets):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)

        cue.set_state(
            V0=action['V0'],
            phi=action['phi'],
            theta=action['theta'],
            a=action['a'],
            b=action['b']
        )

        if simulate_with_timeout(shot):
            return analyze_shot_for_reward(shot, balls, my_targets)
        return -1000

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()

        # 1. 启发式生成
        candidates = self.generate_heuristic_actions(balls, my_targets, table)

        # 2. 快速筛选最佳候选
        best_candidate = None
        best_score = -float('inf')

        for action in candidates:
            score = self._simulate_and_score(action, balls, table, my_targets)
            if score > best_score:
                best_score = score
                best_candidate = action

        if best_candidate is None or best_score < -100:
             return self._random_action()

        # 3. 贝叶斯优化微调
        # print(f"[NewAgent] Refining best heuristic: {best_candidate} (Score: {best_score})")

        phi_center = best_candidate['phi']

        # 定义搜索范围
        pbounds = {
            'V0': (max(0.5, best_candidate['V0'] - 0.5), min(8.0, best_candidate['V0'] + 0.5)),
            'phi': (phi_center - 2.0, phi_center + 2.0), # 小范围微调角度
            'theta': (0, 0), # 简化，暂不优化theta
            'a': (max(-0.5, best_candidate['a'] - 0.1), min(0.5, best_candidate['a'] + 0.1)),
            'b': (max(-0.5, best_candidate['b'] - 0.1), min(0.5, best_candidate['b'] + 0.1))
        }

        def reward_function(V0, phi, theta, a, b):
            action = {'V0': V0, 'phi': phi % 360, 'theta': theta, 'a': a, 'b': b}
            return self._simulate_and_score(action, balls, table, my_targets)

        try:
            optimizer = BayesianOptimization(
                f=reward_function,
                pbounds=pbounds,
                random_state=np.random.randint(1000),
                verbose=0
            )

            optimizer.maximize(init_points=self.bo_init_points, n_iter=self.bo_n_iter)

            best_res = optimizer.max
            refined_action = best_res['params']
            refined_action['phi'] = refined_action['phi'] % 360

            # print(f"[NewAgent] Refined action: {refined_action} (Score: {best_res['target']})")
            return refined_action
        except Exception as e:
            print(f"[NewAgent] BO failed: {e}, using heuristic.")
            return best_candidate
