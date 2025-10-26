# -*- coding: utf-8 -*-
"""
基于 Pyomo 的电力系统最优潮流求解器（面向对象版本）
作者: Chao Shen
日期: 2025
"""

from pyomo.environ import *
from pyomo.opt import SolverFactory
import math
import numpy as np
from pypower.api import case14, case30, case57, case118, case300, loadcase
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from pypower.idx_brch import QT
from numpy import c_, zeros


class PowerSystemOPF:
    """
    电力系统最优潮流求解器类
    
    功能:
        1. 加载 PyPower 格式的电力系统数据
        2. 转换数据为 Pyomo 优化模型所需格式
        3. 构建和求解优化问题
        4. 提供结果分析和可视化
    """
    
    def __init__(self, case_name='case118', verbose=True):
        """
        初始化求解器
        
        参数:
            case_name: str, 系统名称或文件路径
            verbose: bool, 是否打印详细信息
        """
        self.case_name = case_name
        self.verbose = verbose
        
        # 原始数据
        self.ppc = None
        self.baseMVA = None
        self.bus = None
        self.gen = None
        self.branch = None
        
        # 处理后的数据
        self.Ybus = None
        self.Ybus_dict = None
        self.line_list = []
        self.type_list = None
        
        # 发电机参数
        self.pg_max_list = None
        self.pg_min_list = None
        self.qg_max_list = None
        self.qg_min_list = None
        
        # 成本参数
        self.cost_dict = None
        
        # Pyomo 模型
        self.model = None
        
        # 求解结果
        self.results = None
        self.PG = None
        self.QG = None
        self.V = None
        self.delta = None
        
        # 加载系统
        self._load_system()
    
    def _print(self, message):
        """打印信息（如果 verbose=True）"""
        if self.verbose:
            print(message)
    
    def _load_system(self):
        """
        步骤1: 加载电力系统数据
        """
        self._print("="*60)
        self._print(f"步骤1: 加载系统 - {self.case_name}")
        self._print("="*60)
        
        # 根据名称加载不同的case
        case_map = {
            'case14': case14,
            'case30': case30,
            'case57': case57,
            'case118': case118,
            'case300': case300
        }
        
        if self.case_name in case_map:
            self.ppc = loadcase(case_map[self.case_name]())
        else:
            # 自定义文件路径
            self.ppc = loadcase(self.case_name)
        
        # 转换为内部编号
        self.ppc = ext2int(self.ppc)
        
        # 提取基本数据
        self.baseMVA = self.ppc["baseMVA"]
        self.bus = self.ppc["bus"]
        self.gen = self.ppc["gen"]
        self.branch = self.ppc["branch"]
        
        # 填充 branch 矩阵
        if self.branch.shape[1] < QT:
            self.branch = c_[self.branch, 
                           zeros((self.branch.shape[0], 
                                 QT - self.branch.shape[1] + 1))]
            self.ppc["branch"] = self.branch
        
        self._print(f"✓ 节点数: {self.bus.shape[0]}")
        self._print(f"✓ 发电机数: {self.gen.shape[0]}")
        self._print(f"✓ 支路数: {self.branch.shape[0]}")
        self._print(f"✓ 基准容量: {self.baseMVA} MVA")
    
    def _process_admittance_matrix(self):
        """
        步骤2: 处理导纳矩阵
        """
        self._print("\n步骤2: 构建导纳矩阵")
        
        # 构建节点导纳矩阵
        self.Ybus, Yf, Yt = makeYbus(self.baseMVA, self.bus, self.branch)
        
        # 转换为字典格式（Pyomo 需要）
        self.Ybus_dict = dict(self.Ybus.todok().items())
        
        self._print(f"✓ 导纳矩阵维度: {self.Ybus.shape}")
        self._print(f"✓ 非零元素数: {self.Ybus.nnz}")
    
    def _process_branch_data(self):
        """
        步骤3: 处理支路数据
        """
        self._print("\n步骤3: 处理支路数据")
        
        # 构建支路列表 (from_bus, to_bus)
        self.line_list = []
        for i in range(self.branch.shape[0]):
            from_bus = int(self.branch[i, 0])
            to_bus = int(self.branch[i, 1])
            self.line_list.append((from_bus, to_bus))
        
        self._print(f"✓ 支路列表长度: {len(self.line_list)}")
    
    def _process_generator_data(self):
        """
        步骤4: 处理发电机数据
        """
        self._print("\n步骤4: 处理发电机数据")
        
        row_bus = self.bus.shape[0]
        row_gen = self.gen.shape[0]
        
        # 初始化发电机参数数组
        self.pg_max_list = np.zeros(row_bus)
        self.pg_min_list = np.zeros(row_bus)
        self.qg_max_list = np.zeros(row_bus)
        self.qg_min_list = np.zeros(row_bus)
        
        # 将发电机参数映射到节点
        for i in range(row_gen):
            gen_bus = int(self.gen[i, 0])
            self.pg_max_list[gen_bus] = self.gen[i, 8] / self.baseMVA
            self.pg_min_list[gen_bus] = self.gen[i, 9] / self.baseMVA
            self.qg_max_list[gen_bus] = self.gen[i, 3] / self.baseMVA
            self.qg_min_list[gen_bus] = self.gen[i, 4] / self.baseMVA
        
        # 获取节点类型
        self.type_list = self.bus[:, 1]
        
        self._print(f"✓ PV 节点数: {np.sum(self.type_list == 2)}")
        self._print(f"✓ 平衡节点数: {np.sum(self.type_list == 3)}")
        self._print(f"✓ PQ 节点数: {np.sum(self.type_list == 1)}")
    
    def _process_cost_data(self):
        """
        步骤5: 处理成本数据
        """
        self._print("\n步骤5: 处理成本数据")
        
        row_bus = self.bus.shape[0]
        gencost_matrix = self.ppc['gencost']
        row_cost, col_cost = gencost_matrix.shape
        
        # 构建节点-成本矩阵
        bus_gen_matrix = np.zeros((row_bus, col_cost))
        gen_order_list = self.gen[:, 0]
        
        for i in range(len(gen_order_list)):
            bus_idx = int(gen_order_list[i])
            bus_gen_matrix[bus_idx, :] = gencost_matrix[i, :]
        
        # 转换为字典
        self.cost_dict = {}
        for i in range(row_bus):
            for j in range(col_cost):
                self.cost_dict[(i, j)] = bus_gen_matrix[i, j]
        
        self._print(f"✓ 成本矩阵维度: {bus_gen_matrix.shape}")
    
    def process_data(self):
        """
        执行所有数据处理步骤
        """
        self._process_admittance_matrix()
        self._process_branch_data()
        self._process_generator_data()
        self._process_cost_data()
        self._print("\n" + "="*60)
        self._print("✓ 数据处理完成")
        self._print("="*60)
    
    def build_model(self, objective_type='loss', v_bounds=(0.95, 1.08), 
                   delta_bounds=(-math.pi/4, math.pi/4)):
        """
        构建 Pyomo 优化模型
        
        参数:
            objective_type: str, 目标函数类型
                           'loss' - 网损最小化
                           'cost' - 发电成本最小化
            v_bounds: tuple, 电压幅值边界 (min, max)
            delta_bounds: tuple, 相角边界 (min, max)
        """
        self._print("\n" + "="*60)
        self._print("步骤6: 构建 Pyomo 优化模型")
        self._print("="*60)
        
        row_bus = self.bus.shape[0]
        col_cost = self.ppc['gencost'].shape[1]
        
        # 创建模型
        self.model = ConcreteModel()
        
        # ========== 定义集合 ==========
        self._print("\n定义集合...")
        self.model.buses = Set(initialize=self.bus[:, 0].astype(int))
        self.model.lines = Set(initialize=self.line_list)
        self.model.cost_dims = Set(initialize=range(col_cost))
        
        # ========== 定义参数 ==========
        self._print("定义参数...")
        self.model.PD = Param(self.model.buses, 
                             initialize=self.bus[:, 2]/self.baseMVA, 
                             default=0.0)
        self.model.QD = Param(self.model.buses, 
                             initialize=self.bus[:, 3]/self.baseMVA, 
                             default=0.0)
        self.model.PG_MAX = Param(self.model.buses, 
                                 initialize=self.pg_max_list, 
                                 default=0.0)
        self.model.PG_MIN = Param(self.model.buses, 
                                 initialize=self.pg_min_list, 
                                 default=0.0)
        self.model.QG_MAX = Param(self.model.buses, 
                                 initialize=self.qg_max_list, 
                                 default=0.0)
        self.model.QG_MIN = Param(self.model.buses, 
                                 initialize=self.qg_min_list, 
                                 default=0.0)
        self.model.Y = Param(self.model.buses * self.model.buses, 
                            initialize=self.Ybus_dict, 
                            default=0.0)
        self.model.C = Param(self.model.buses * self.model.cost_dims, 
                            initialize=self.cost_dict, 
                            default=0.0)
        
        # ========== 定义变量 ==========
        self._print("定义变量...")
        self.model.PG = Var(self.model.buses, domain=Reals)
        self.model.QG = Var(self.model.buses, domain=Reals)
        self.model.V = Var(self.model.buses, initialize=1.0, bounds=v_bounds)
        self.model.delta = Var(self.model.buses, initialize=0, bounds=delta_bounds)
        
        # 设置变量边界和固定值
        for i in range(row_bus):
            if self.type_list[i] in [2, 3]:  # PV 或平衡节点
                self.model.PG[i].setlb(self.pg_min_list[i])
                self.model.PG[i].setub(self.pg_max_list[i])
                self.model.QG[i].setlb(self.qg_min_list[i])
                self.model.QG[i].setub(self.qg_max_list[i])
                
                if self.type_list[i] == 3:  # 平衡节点
                    self.model.delta[i].fix(0)
                    self.model.V[i].fix(1.0)
            else:  # PQ 节点
                self.model.PG[i].fix(0)
                self.model.QG[i].fix(0)
        
        # ========== 定义约束 ==========
        self._print("定义约束...")
        self._add_branch_constraints()
        self._add_power_balance_constraints()
        self._add_generator_constraints()
        
        # ========== 定义目标函数 ==========
        self._print("定义目标函数...")
        self._add_objective(objective_type)
        
        self._print("\n✓ 模型构建完成")
        self._print(f"  变量数: {len([v for v in self.model.component_data_objects(Var)])}")
        self._print(f"  约束数: {len([c for c in self.model.component_data_objects(Constraint)])}")
    
    def _add_branch_constraints(self):
        """添加支路潮流限制约束"""
        self.model.branch_limits = ConstraintList()
        MVA_list = self.branch[:, 5]
        
        for i in range(len(MVA_list)):
            line_limit = (MVA_list[i] / self.baseMVA) ** 2
            f_bus = int(self.branch[i, 0])
            t_bus = int(self.branch[i, 1])
            
            # 计算支路阻抗
            Z_f_t = -1 / self.model.Y[f_bus, t_bus]
            R_f_t = Z_f_t.real
            X_f_t = Z_f_t.imag
            
            # 计算电压的实部和虚部
            U_f_real = self.model.V[f_bus] * cos(self.model.delta[f_bus])
            U_f_imag = self.model.V[f_bus] * sin(self.model.delta[f_bus])
            U_t_real = self.model.V[t_bus] * cos(self.model.delta[t_bus])
            U_t_imag = self.model.V[t_bus] * sin(self.model.delta[t_bus])
            
            # 电压差
            dif_U_real = U_f_real - U_t_real
            dif_U_imag = U_f_imag - U_t_imag
            
            # 电流
            I_f_t_real = (dif_U_real * R_f_t + X_f_t * dif_U_imag) / (R_f_t**2 + X_f_t**2)
            I_f_t_imag = (dif_U_imag * R_f_t - X_f_t * dif_U_real) / (R_f_t**2 + X_f_t**2)
            
            # 功率
            P_f_t = U_f_real * I_f_t_real - U_f_imag * I_f_t_imag
            Q_f_t = U_f_imag * I_f_t_real + U_f_real * I_f_t_imag
            
            # 视在功率平方
            S_f_t_2 = P_f_t**2 + Q_f_t**2
            
            # 添加约束
            self.model.branch_limits.add(expr=S_f_t_2 <= line_limit)
    
    def _add_power_balance_constraints(self):
        """添加功率平衡约束"""
        self.model.power_balance_constraints = ConstraintList()
        
        for i in self.model.buses:
            # 节点注入有功功率
            real_power_inj = sum(
                self.model.V[i] * self.model.V[j] * (
                    self.model.Y[i, j].real * cos(self.model.delta[i] - self.model.delta[j]) +
                    self.model.Y[i, j].imag * sin(self.model.delta[i] - self.model.delta[j])
                ) for j in self.model.buses
            )
            
            # 节点注入无功功率
            reactive_power_inj = sum(
                self.model.V[i] * self.model.V[j] * (
                    self.model.Y[i, j].real * sin(self.model.delta[i] - self.model.delta[j]) -
                    self.model.Y[i, j].imag * cos(self.model.delta[i] - self.model.delta[j])
                ) for j in self.model.buses
            )
            
            # 根据节点类型添加约束
            if self.type_list[i] in [2, 3]:  # PV 或平衡节点
                self.model.power_balance_constraints.add(
                    expr=self.model.PG[i] - self.model.PD[i] == real_power_inj
                )
                self.model.power_balance_constraints.add(
                    expr=self.model.QG[i] - self.model.QD[i] == reactive_power_inj
                )
            elif self.type_list[i] == 1:  # PQ 节点
                self.model.power_balance_constraints.add(
                    expr=-self.model.PD[i] == real_power_inj
                )
                self.model.power_balance_constraints.add(
                    expr=-self.model.QD[i] == reactive_power_inj
                )
    
    def _add_generator_constraints(self):
        """添加发电机出力限制约束"""
        self.model.generator_limits = ConstraintList()
        
        for i in self.model.buses:
            if self.type_list[i] in [2, 3]:
                self.model.generator_limits.add(
                    expr=self.model.PG[i] >= self.model.PG_MIN[i]
                )
                self.model.generator_limits.add(
                    expr=self.model.PG[i] <= self.model.PG_MAX[i]
                )
                self.model.generator_limits.add(
                    expr=self.model.QG[i] >= self.model.QG_MIN[i]
                )
                self.model.generator_limits.add(
                    expr=self.model.QG[i] <= self.model.QG_MAX[i]
                )
    
    def _add_objective(self, objective_type):
        """添加目标函数"""
        if objective_type == 'loss':
            # 网损最小化
            self.model.obj = Objective(
                expr=(sum(self.model.PG[i] for i in self.model.buses) - 
                     sum(self.model.PD[i] for i in self.model.buses))**2,
                sense=minimize
            )
        elif objective_type == 'cost':
            # 发电成本最小化
            self.model.obj = Objective(
                expr=sum(
                    (self.model.PG[i] * self.baseMVA)**2 * self.model.C[i, 4] +
                    (self.model.PG[i] * self.baseMVA) * self.model.C[i, 5] +
                    self.model.C[i, 6] + self.model.C[i, 1]
                    for i in self.model.buses
                ),
                sense=minimize
            )
        else:
            raise ValueError(f"未知的目标函数类型: {objective_type}")
    
    def solve(self, solver_name='ipopt', tee=True, solver_options=None):
        """
        求解优化问题
        
        参数:
            solver_name: str, 求解器名称
            tee: bool, 是否显示求解过程
            solver_options: dict, 求解器选项
        """
        self._print("\n" + "="*60)
        self._print("步骤7: 求解优化问题")
        self._print("="*60)
        
        if self.model is None:
            raise ValueError("模型未构建，请先调用 build_model()")
        
        # 创建求解器
        solver = SolverFactory(solver_name)
        
        # 设置求解器选项
        if solver_options:
            for key, value in solver_options.items():
                solver.options[key] = value
        
        # 求解
        self.results = solver.solve(self.model, tee=tee)
        
        # 提取结果
        self._extract_results()
        
        self._print("\n✓ 求解完成")
        self._print(f"  状态: {self.results.solver.status}")
        self._print(f"  终止条件: {self.results.solver.termination_condition}")
    
    def _extract_results(self):
        """提取求解结果"""
        row_bus = self.bus.shape[0]
        
        self.PG = [self.model.PG[i].value for i in range(row_bus)]
        self.QG = [self.model.QG[i].value for i in range(row_bus)]
        self.V = [self.model.V[i].value for i in range(row_bus)]
        self.delta = [self.model.delta[i].value for i in range(row_bus)]
    
    def print_results(self):
        """打印求解结果"""
        if self.PG is None:
            print("没有可用的求解结果")
            return
        
        print("\n" + "="*60)
        print("求解结果摘要")
        print("="*60)
        
        # 计算关键指标
        PD = [self.model.PD[i] for i in range(len(self.PG))]
        total_gen = sum(self.PG)
        total_load = sum(PD)
        loss = total_gen - total_load
        loss_rate = (loss / total_load) * 100 if total_load > 0 else 0
        
        print(f"\n【功率平衡】")
        print(f"  总发电量: {total_gen:.4f} p.u. ({total_gen * self.baseMVA:.2f} MW)")
        print(f"  总负荷:   {total_load:.4f} p.u. ({total_load * self.baseMVA:.2f} MW)")
        print(f"  网损:     {loss:.4f} p.u. ({loss * self.baseMVA:.2f} MW)")
        print(f"  网损率:   {loss_rate:.2f}%")
        
        print(f"\n【电压水平】")
        print(f"  最大电压: {max(self.V):.4f} p.u.")
        print(f"  最小电压: {min(self.V):.4f} p.u.")
        print(f"  平均电压: {np.mean(self.V):.4f} p.u.")
        
        print(f"\n【相角范围】")
        delta_deg = [d * 180 / math.pi for d in self.delta]
        print(f"  最大相角: {max(delta_deg):.2f}°")
        print(f"  最小相角: {min(delta_deg):.2f}°")
        
        print(f"\n【发电机出力】")
        gen_count = 0
        for i in range(len(self.PG)):
            if self.type_list[i] in [2, 3]:
                print(f"  Bus {i}: P = {self.PG[i]:.4f} p.u., Q = {self.QG[i]:.4f} p.u.")
                gen_count += 1
        print(f"  (共 {gen_count} 台发电机)")
        
        if self.model.obj is not None:
            print(f"\n【目标函数值】")
            print(f"  {self.model.obj():.6f}")
    
    def get_results_dict(self):
        """
        返回结果字典
        
        返回:
            dict: 包含所有求解结果的字典
        """
        if self.PG is None:
            return None
        
        PD = [self.model.PD[i] for i in range(len(self.PG))]
        
        return {
            'PG': self.PG,
            'QG': self.QG,
            'V': self.V,
            'delta': self.delta,
            'PD': PD,
            'total_generation': sum(self.PG),
            'total_load': sum(PD),
            'loss': sum(self.PG) - sum(PD),
            'loss_rate': ((sum(self.PG) - sum(PD)) / sum(PD)) * 100,
            'v_max': max(self.V),
            'v_min': min(self.V),
            'objective_value': self.model.obj() if hasattr(self.model, 'obj') else None
        }
    
    def save_results(self, filename='opf_results.txt'):
        """
        保存结果到文件
        
        参数:
            filename: str, 输出文件名
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"电力系统最优潮流求解结果\n")
            f.write(f"系统: {self.case_name}\n")
            f.write("="*60 + "\n\n")
            
            results = self.get_results_dict()
            
            f.write("【功率平衡】\n")
            f.write(f"总发电量: {results['total_generation']:.4f} p.u.\n")
            f.write(f"总负荷: {results['total_load']:.4f} p.u.\n")
            f.write(f"网损: {results['loss']:.4f} p.u.\n")
            f.write(f"网损率: {results['loss_rate']:.2f}%\n\n")
            
            f.write("【节点电压】\n")
            for i in range(len(self.V)):
                f.write(f"Bus {i}: {self.V[i]:.4f} p.u.\n")
            
            f.write("\n【节点相角】\n")
            for i in range(len(self.delta)):
                f.write(f"Bus {i}: {self.delta[i]*180/math.pi:.2f}°\n")
            
            f.write("\n【发电机出力】\n")
            for i in range(len(self.PG)):
                if self.type_list[i] in [2, 3]:
                    f.write(f"Bus {i}: P = {self.PG[i]:.4f} p.u., Q = {self.QG[i]:.4f} p.u.\n")
        
        self._print(f"\n✓ 结果已保存到: {filename}")
    
    def print_model_structure(self):
        """打印模型结构"""
        if self.model is None:
            print("模型未构建")
            return
        
        print("\n" + "="*60)
        print("模型结构")
        print("="*60)
        self.model.pprint()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    
    # 示例1: 基本使用
    print("="*80)
    print("示例1: 求解 IEEE 118 节点系统（网损最优）")
    print("="*80)
    
    # 创建求解器实例
    opf = PowerSystemOPF(case_name='case118', verbose=True)
    
    # 处理数据
    opf.process_data()
    
    # 构建模型
    opf.build_model(objective_type='loss', v_bounds=(0.95, 1.08))
    
    # 求解
    opf.solve(solver_name='ipopt', tee=True)
    
    # 打印结果
    opf.print_results()
    
    # 保存结果
    opf.save_results('case118_loss_results.txt')
    
    # 获取结果字典
    results = opf.get_results_dict()
    print(f"\n网损率: {results['loss_rate']:.2f}%")
    
    
    # 示例2: 批量测试多个系统
    print("\n\n" + "="*80)
    print("示例2: 批量测试多个系统")
    print("="*80)
    
    cases = ['case14', 'case30', 'case57', 'case118']
    
    for case in cases:
        print(f"\n{'='*60}")
        print(f"测试 {case}")
        print('='*60)
        
        try:
            opf = PowerSystemOPF(case_name=case, verbose=False)
            opf.process_data()
            opf.build_model(objective_type='loss')
            opf.solve(tee=False)
            opf.print_results()
        except Exception as e:
            print(f"✗ {case} 求解失败: {e}")
    
    
    # 示例3: 经济调度（成本最优）
    print("\n\n" + "="*80)
    print("示例3: IEEE 57 节点系统（成本最优）")
    print("="*80)
    
    opf_cost = PowerSystemOPF(case_name='case57', verbose=True)
    opf_cost.process_data()
    opf_cost.build_model(objective_type='cost', v_bounds=(0.95, 1.08))
    opf_cost.solve(solver_name='ipopt', tee=False)
    opf_cost.print_results()