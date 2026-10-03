# 精确 Gibbs 排列传输与 Energy-KL 投影

## 1. 问题定义

源预测器输出有限概率测度

\[
Q=\sum_{i=1}^{K}q_i\delta_{s_i},\qquad q\in\Delta_K^\circ,
\]

互补预测器输出目标候选轨迹
\(T=(t_1,\ldots,t_K)\)、目标无关的预测风险
\(r\in\mathbb R^K\)及固定的单轨迹决策。概率前向过程不使用真实未来轨迹。

源候选轨迹 \(i\) 与目标候选轨迹 \(j\) 的跨支持代价为

\[
C_{ij}=\frac{\operatorname{ADE}(s_i,t_j)}{a}
       +\frac{\operatorname{FDE}(s_i,t_j)}{f},
\]

其中，\(a\) 和 \(f\) 是仅由训练数据确定的固定尺度。对排列
\(\sigma\in\mathfrak S_K\)，定义一般形式的分配代价

\[
c_w(\sigma)=\sum_i w_i C_{i,\sigma(i)},\qquad w\in\Delta_K.
\]

最终选择的算法使用均匀分配权重 \(w_i=1/K\)，而不是
\(w_i=q_i\)。源概率 \(q\) 仍作为被传输的概率质量参与后续边缘化：

\[
\rho_w(\sigma)=\frac{\exp[-c_w(\sigma)]}
 {\sum_{\tau\in\mathfrak S_K}\exp[-c_w(\tau)]},\qquad
u_j=\sum_{\sigma}\rho_w(\sigma)
       \sum_iq_i\mathbf 1\{\sigma(i)=j\}.
\]

目标支持上的平均轨迹距离矩阵为

\[
D_{ij}=\frac{1}{H}\sum_{h=1}^{H}\lVert t_{ih}-t_{jh}\rVert_2.
\]

算子返回下列严格凸问题的唯一解：

\[
p^*=\arg\min_{p\in\Delta_K}
  p^\top r-\frac12p^\top Dp+\operatorname{KL}(p\Vert u).
\]

传输和投影不包含可训练参数、残差修正、门控或测试标签。正式概率对照在开发集上预先选择了标量概率温度 \(T_p=0.75\)，并在测试推理前冻结。该温度属于校准配置，不改变上述算子的定义，也不构成独立方法创新。

## 2. 算法

```text
输入：源候选轨迹 S、源概率 q、目标候选轨迹 T、预测风险 r
1. 计算 K x K 跨支持代价矩阵 C。
2. 枚举全部源到目标的一一对应排列 sigma。
3. 以均匀分配权重计算 c_w(sigma)，并得到 Gibbs 分布 rho_w。
4. 通过 rho_w 对源概率 q 进行边缘化，得到目标先验 u。
5. 计算目标支持上的成对距离矩阵 D。
6. 采用带确定性 Armijo 回溯的等式约束 Newton 法求解 Energy-KL 问题。
输出：目标支持 T、概率 p* 和目标预测器的固定决策。
```

支持代价已知后，精确枚举的单样本复杂度为 \(O(K!K)\)。该实现仅在正式实验的 \(K=5\) 小基数设置下使用。当前 top-M 实现仍先完成全排列枚举再截断，因此只能用于近似误差对照，不能作为可扩展算法。固定迭代 Sinkhorn 算法是近似对照，不是本文算子的等价实现。

## 3. Gibbs 变分表征

令 \(U\) 为 \(\mathfrak S_K\) 上的均匀分布，则 \(\rho_w\) 是下式的唯一最优解：

\[
\min_{\rho\in\Delta_{K!}}
  \mathbb E_{\sigma\sim\rho}[c_w(\sigma)]
  +\operatorname{KL}(\rho\Vert U).
\]

对约束 \(\sum_\sigma\rho(\sigma)=1\) 引入乘子后，一阶条件给出
\(\log\rho(\sigma)=-c_w(\sigma)+\text{constant}\)。排列单纯形上的 Kullback-Leibler（KL）散度严格凸，因此最优解唯一。该结论说明算子对对应关系不确定性进行边缘化，而不是选择单一硬对应。

## 4. 传输概率位于单纯形内部

对任一排列，被映射向量 \(q^\top P_\sigma\) 是 \(q\) 的排列，因而属于
\(\Delta_K^\circ\)。有限代价下每个排列的 Gibbs 概率均为正，故其凸组合 \(u\) 满足

\[
u_j>0\quad\text{且}\quad\sum_j u_j=1.
\]

这里的 \(q\) 是被传输的源概率，而不是最终算法的分配代价权重。二者必须在论文中分别表述。

## 5. 凸性、存在性与唯一性

对满足 \(\mathbf 1^\top z=0\) 的向量 \(z\)，欧氏距离是条件负定核：

\[
\sum_{ij}z_i z_j\lVert x_i-x_j\rVert_2\le 0.
\]

因此，\(-\tfrac12p^\top Dp\) 在概率单纯形的仿射子空间上为凸函数；风险项为线性函数，且

\[
\nabla^2\operatorname{KL}(p\Vert u)=\operatorname{diag}(1/p_i)
\]

在相对内部正定。完整目标函数严格凸且下半连续，定义域为紧集，故最优解存在且唯一。实现使用 FP64 求解一阶条件，并以 Karush-Kuhn-Tucker（KKT）残差和切空间 Hessian 数值测试作为补充验证。

## 6. 源置换不变性与目标置换等变性

设 \(A\) 和 \(B\) 分别重标记源候选轨迹和目标候选轨迹，则
\(q'=Aq\)、\(S'=AS\)、\(T'=BT\)、\(r'=Br\) 且
\(C'=ACB^\top\)。均匀分配权重在源重标记后保持不变，排列集合之间存在双射，因此 Gibbs 概率只发生重索引，传输先验满足 \(u'=Bu\)。又因
\(D'=BDB^\top\)，投影目标满足 \(F'(Bp)=F(p)\)。由唯一性可得
\(p^{*\prime}=Bp^*\)。

## 7. 主张边界

上述论证只建立算子的良定性、唯一性和重标记性质。它不证明候选轨迹生成器具有域外泛化能力，也不证明精确枚举可扩展至大基数，或该方法对所有概率评分均占优。质量加权分配仅作为历史消融保留；开发集最终选择记录判定其独立贡献未成立，因此不得将其写为最终算法组成或创新点。跨不同支持比较概率质量时，应使用固定事件负对数似然、Brier 分数或 Energy Score，不得用支持索引依赖的模式标签分数替代。
