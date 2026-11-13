# Integrated Decision and Control: Towards Interpretable and Computationally Efficient Driving Intelligence

# 集成决策与控制:迈向可解释且计算高效的驾驶智能

Yang Guan \( {}^{1} \) , Yangang Ren \( {}^{1} \) , Qi Sun \( {}^{1} \) , Shengbo Eben Li*1, Haitong Ma \( {}^{1} \) , Jingliang Duan \( {}^{1} \) , Yifan Dai \( {}^{2} \) , Bo Cheng \( {}^{1} \)

阳关\( {}^{1} \)，任彦刚\( {}^{1} \)，孙琦\( {}^{1} \)，李升波*1，马海通\( {}^{1} \)，段景亮\( {}^{1} \)，戴一凡\( {}^{2} \)，程博\( {}^{1} \)

Abstract-Decision and control are core functionalities of high-level automated vehicles. Current mainstream methods, such as functionality decomposition and end-to-end reinforcement learning (RL), either suffer high time complexity or poor interpretability and adaptability on real-world autonomous driving tasks. In this paper, we present an interpretable and computationally efficient framework called integrated decision and control (IDC) for automated vehicles, which decomposes the driving task into static path planning and dynamic optimal tracking that are structured hierarchically. First, the static path planning generates several candidate paths only considering static traffic elements. Then, the dynamic optimal tracking is designed to track the optimal path while considering the dynamic obstacles. To that end, we formulate a constrained optimal control problem (OCP) for each candidate path, optimize them separately and follow the one with the best tracking performance. To unload the heavy online computation, we propose a model-based reinforcement learning (RL) algorithm that can be served as an approximate constrained OCP solver. Specifically, the OCPs for all paths are considered together to construct a single complete RL problem and then solved offline in the form of value and policy networks, for real-time online path selecting and tracking respectively. We verify our framework in both simulations and the real world. Results show that compared with baseline methods IDC has an order of magnitude higher online computing efficiency, as well as better driving performance including traffic efficiency and safety. In addition, it yields great interpretability and adaptability among different driving tasks. The effectiveness of the proposed method is also demonstrated in real road tests with complicated traffic conditions.

摘要——决策与控制是高级自动驾驶车辆的核心功能。当前的主流方法，如功能分解和端到端强化学习(RL)，在实际自动驾驶任务中要么时间复杂度高，要么可解释性和适应性差。在本文中，我们提出了一种用于自动驾驶车辆的可解释且计算高效的框架，称为集成决策与控制(IDC)，它将驾驶任务分解为静态路径规划和动态最优跟踪，并进行分层结构设计。首先，静态路径规划仅考虑静态交通元素生成多条候选路径。然后，动态最优跟踪旨在在考虑动态障碍物的同时跟踪最优路径。为此，我们为每条候选路径制定一个约束最优控制问题(OCP)，分别对其进行优化，并选择跟踪性能最佳的路径。为了减轻繁重的在线计算负担，我们提出了一种基于模型的强化学习(RL)算法，它可以作为一个近似的约束OCP求解器。具体来说，将所有路径的OCP一起考虑，构建一个完整的RL问题，然后以价值网络和策略网络的形式离线求解，分别用于实时在线路径选择和跟踪。我们在模拟和实际场景中验证了我们的框架。结果表明，与基线方法相比，IDC的在线计算效率提高了一个数量级，并且具有更好的驾驶性能，包括交通效率和安全性。此外，它在不同驾驶任务中具有很强的可解释性和适应性。所提方法的有效性也在复杂交通条件下的实际道路测试中得到了证明。

Index Terms-Automated vehicle, Decision and control, Reinforcement learning, Model-based.

关键词——自动驾驶车辆，决策与控制，强化学习，基于模型的

## I. INTRODUCTION

## I. 引言

Intelligence of automobile technology and driving assistance system has great potential to improve safety, reduce fuel consumption and enhance traffic efficiency, which will completely change the road transportation. Decision and control are indispensable for high-level autonomous driving, which are in charge of computing the expected instructions of steering and acceleration relying on the environment perception results. It is generally believed that there are two technical routes for the decision and control of automated vehicles: decomposed scheme and end-to-end scheme.

汽车技术和驾驶辅助系统的智能化在提高安全性、降低油耗和提升交通效率方面具有巨大潜力，这将彻底改变道路运输。决策与控制对于高级自动驾驶不可或缺，它们负责根据环境感知结果计算转向和加速的预期指令。一般认为，自动驾驶车辆的决策与控制有两种技术路线:分解式方案和端到端方案。

Decomposed scheme splits the decision and control functionality into several serial submodules, such as prediction, behavior selection, trajectory planning and control [1]. Prediction is to predict the future trajectory of traffic participants to determine the feasible region in future time steps [2]. It is further decomposed into behavior recognition [3], [4] and trajectory prediction [5], [6]. Since the prediction algorithms usually works on each surrounding vehicle, it means that the more the number of vehicles, the more computation is needed. Behavior selection is then used to choose a high-level driving behavior relying on an expert system in which many designed rules are embedded. Typical methods include finite state machine [7] and decision tree [8]. Based on the selected behavior, a collision free space-time curve satisfying vehicle dynamics is calculated according to the predicted trajectories and road constraints by the trajectory planning submodule. Three main categories of the planning algorithms include optimization-based, search-based and sample-based. Optimization-based methods formulate the planning problem into an optimization problem, where specific aspects of trajectory are optimized and constraints are considered [9], [10]. However, it suffers from long computational time for large-scale nonlinear and non-convex problem. The search-based methods represented by A* and Rapidly-exploring Random Tree are more efficient [11]-[16], but they usually lead to low-resolution paths and can barely take dynamic obstacles into consideration. The sample-based methods also have poor computing efficiency because they sample points and interpolate them evenly in the whole state space [17], [18]. Xin et al. proposed a combination of the optimization-based and search-based methods, where a trajectory is searched in the space-time by A* and then smoothed by model predictive control (MPC), yielding the best performance in terms of planning time and comfort [19]. Finally, the controller is used to follow the planned trajectory and calculate the expected controls by linear quadratic regulator or MPC [20], [21]. The decomposed scheme requires large amount of human design but is still hard to cover all possible driving scenarios due to the long tail effect. Besides, the real time performance cannot be guaranteed because it is time-consuming to complete all the works serially in a limited time for industrial computers.

分解式方案将决策与控制功能拆分为几个串行子模块，如预测、行为选择、轨迹规划和控制[1]。预测是预测交通参与者的未来轨迹，以确定未来时间步的可行区域[2]。它进一步分解为行为识别[3,4]和轨迹预测[5,6]。由于预测算法通常对每辆周围车辆进行运算，这意味着车辆数量越多，所需计算量越大。行为选择随后用于依靠嵌入许多设计规则的专家系统选择高级驾驶行为。典型方法包括有限状态机[7]和决策树[8]。基于选定的行为，轨迹规划子模块根据预测轨迹和道路约束计算出一条满足车辆动力学的无碰撞时空曲线。规划算法主要有三类:基于优化的、基于搜索的和基于采样的。基于优化的方法将规划问题表述为一个优化问题，在其中优化轨迹的特定方面并考虑约束条件[9,10]。然而，对于大规模非线性和非凸问题，其计算时间较长。以A*和快速探索随机树为代表的基于搜索的方法效率更高[11 - 16]，但它们通常导致路径分辨率低，并且几乎无法考虑动态障碍物。基于采样的方法计算效率也很差，因为它们在整个状态空间中对采样点进行均匀插值[17,18]。Xin等人提出了一种基于优化和基于搜索方法的组合，其中通过A*在时空上搜索一条轨迹，然后通过模型预测控制(MPC)进行平滑处理，在规划时间和舒适性方面取得了最佳性能[19]。最后，控制器用于跟踪规划轨迹，并通过线性二次调节器或MPC计算预期控制量[20,21]。分解式方案需要大量人工设计，但由于长尾效应，仍然难以涵盖所有可能的驾驶场景。此外，由于工业计算机在有限时间内串行完成所有工作耗时较长，因此无法保证实时性能。

End-to-end scheme computes the expected instructions directly from inputs given by perception module using a policy usually carried out by a deep neural network (NN). Reinforcement learning (RL) methods do not rely on labelled driving data but learn by trial-and-error in real-world or a high fidelity simulator [22], [23]. Early RL applications on autonomous driving mainly focus on learning a single driving behavior, e.g., lane keeping [24], lane changing [25] or overtaking [26]. They usually employ deep Q-networks 27 or deep deterministic policy gradient method [28] to learn policy in discrete or continuous domain. Besides, they own different reward functions for their respective goals. Recently, RL has been applied in certain driving scenarios. Duan et al. realized decision making under a virtual two-lane highway using hierarchical RL, which designs complicated reward functions for its high-level maneuver selection and three low-level maneuvers respectively. Guan et al. achieved centralized control in a four-leg single-lane intersection with sparse rewards, in which only eight cars are considered [29]. Chen et al. designed a bird-view representation and used visual encoding to capture the low-dimensional latent states, solving the driving task in a roundabout scenario with dense surrounding vehicles in a high-definition driving simulator [30]. However, they reported limited safety performance and poor learning efficiency. Current RL methods mostly work on a specific task, in which a set of complicated reward functions is required to offer guidance for policy optimization, such as, distance travelled towards a destination, collisions with other road users or scene objects, maintaining comfort and stability while avoiding extreme acceleration, braking or steering. It is non-trivial and needs a lot of human efforts to tune, causing poor adaptability among driving scenarios and tasks. Besides, the outcome of the policy is hard to interpret, which makes it barely used in real autonomous driving tasks. Moreover, they cannot deal with safety constraints explicitly and suffer from low convergence speed.

端到端方案使用通常由深度神经网络(NN)执行的策略，直接根据感知模块给出的输入来计算预期指令。强化学习(RL)方法不依赖于标记的驾驶数据，而是通过在现实世界或高保真模拟器中进行试错学习[22],[23]。早期的自动驾驶RL应用主要集中在学习单一驾驶行为，例如车道保持[24]、变道[25]或超车[26]。它们通常采用深度Q网络[27]或深度确定性策略梯度方法[28]在离散或连续域中学习策略。此外，它们针对各自的目标拥有不同的奖励函数。最近，RL已应用于某些驾驶场景。段等人使用分层RL在虚拟双车道高速公路上实现了决策，该方法分别为其高级机动选择和三个低级机动设计了复杂的奖励函数。关等人在稀疏奖励的四岔单车道交叉路口实现了集中控制，其中仅考虑了八辆车[29]。陈等人设计了一种鸟瞰表示，并使用视觉编码来捕获低维潜在状态，在高清驾驶模拟器中解决了周围车辆密集的环形交叉路口场景中的驾驶任务[30]。然而，他们报告了有限的安全性能和较差的学习效率。当前的RL方法大多在特定任务上工作，其中需要一组复杂的奖励函数来为策略优化提供指导，例如，向目的地行驶的距离、与其他道路使用者或场景物体的碰撞、在避免极端加速、制动或转向的同时保持舒适性和稳定性。这并非易事，需要大量人力进行调整，导致在驾驶场景和任务之间的适应性较差。此外，策略的结果难以解释，这使得它几乎无法用于实际的自动驾驶任务。而且，它们无法明确处理安全约束，并且收敛速度较慢。

---

This work is supported by International Science & Technology Cooperation Program of China under 2019YFE0100200, NSF China with 51575293, and U20A20334. It is also partially supported by Tsinghua University-Toyota Joint Research Center for AI Technology of Automated Vehicle.

本工作得到了中国国际科技合作计划(2019YFE0100200)、国家自然科学基金(51575293)以及清华大学-丰田汽车人工智能技术联合研究中心的支持。同时也得到了清华大学车辆与运载学院的部分支持。

\( {}^{1} \) School of Vehicle and Mobility, Tsinghua University, Beijing,100084, China. \( {}^{2} \) Suzhou Automotive Research Institute, Tsinghua University, Suzhou, 215200, China. All correspondence should be sent to S. Eben Li. <lisb04@gmail.com>.

\( {}^{1} \) 清华大学车辆与运载学院，北京 100084，中国。\( {}^{2} \) 清华大学苏州汽车研究院，苏州 215200，中国。所有通信应发送至S. Eben Li。<lisb04@gmail.com>。

---

In this paper, we propose an integrated decision and control framework (IDC) for automated vehicles, which has great interpretability and online computing efficiency, and is applicable in different driving scenarios and tasks. The contributions emphasize in three parts:

在本文中，我们提出了一种用于自动驾驶车辆的集成决策与控制框架(IDC)，它具有很强的可解释性和在线计算效率，适用于不同的驾驶场景和任务。贡献主要体现在三个方面:

1) We proposed an IDC framework for automated vehicles, which decomposes driving tasks into static path planning and dynamic optimal tracking hierarchically. The high-level static path planning is used to generate multiple paths only considering static constraints such as road topology, traffic lights. The low-level dynamic optimal tracking is used to select the optimal path and track it considering dynamic obstacles, wherein a finite-horizon constrained optimal control problem (OCP) is constructed and optimized for each candidate path. The optimal path is selected as the one with the lowest optimal cost function. The IDC framework is computationally efficient because we unload the heavy online optimizations by solving the constrained OCPs offline in the form of value and policy NNs using RL for path selecting and tracking thereafter. It is interpretable in the sense that the solved value and policy functions are the approximation for the optimal cost and the optimal action of the constrained OCP. Moreover, the IDC employs RL to solve a task-independent OCP with tracking errors as objective and safety constraints, making it applicable among a variety of scenarios and tasks.

1)我们提出了一种用于自动驾驶车辆的IDC框架，该框架将驾驶任务分层分解为静态路径规划和动态最优跟踪。高级静态路径规划仅用于生成考虑道路拓扑、交通信号灯等静态约束的多条路径。低级动态最优跟踪用于选择最优路径并考虑动态障碍物进行跟踪，其中为每个候选路径构建并优化一个有限时域约束最优控制问题(OCP)。选择最优成本函数最低的路径作为最优路径。IDC框架计算效率高，因为我们通过使用RL以值和策略神经网络的形式离线求解约束OCP来卸载繁重的在线优化，此后用于路径选择和跟踪。从某种意义上说它是可解释的，即求解的值和策略函数是约束OCP的最优成本和最优动作的近似。此外，IDC采用RL以跟踪误差为目标并考虑安全约束来求解与任务无关的OCP，使其适用于各种场景和任务。

2) We develop a model-based RL algorithm called generalized exterior point method (GEP) for the purpose of approximately solving OCP with large-scale state-wise constraints. The GEP is in fact an extension of the exterior point method in the optimization domain to the field of NN, in which it first constructs an extensive problem involving all the candidate paths and transforms it into an unconstrained problem with a penalty on safety violations. Afterward, the approximate feasible optimal control policy is obtained by alternatively performing gradient descent and enlarging the penalty. The convergence of the GEP is proved. The GEP is the core of IDC because it can deal with a large number of state-wise constraints explicitly and update NNs efficiently with the guidance of model. To the best of our knowledge, GEP is the first model-based solver for OCPs with large-scale state-wise constraints that are parameterized by NNs.

2)我们开发了一种基于模型的RL算法，称为广义外点法(GEP)，用于近似求解具有大规模状态约束的OCP。GEP实际上是优化领域中外点法在神经网络领域的扩展，其中它首先构建一个涉及所有候选路径的扩展问题，并将其转化为一个对违反安全进行惩罚的无约束问题。然后，通过交替执行梯度下降和增大惩罚来获得近似可行的最优控制策略。证明了GEP的收敛性。GEP是IDC的核心，因为它可以明确处理大量的状态约束，并在模型的指导下有效地更新神经网络。据我们所知，GEP是第一个用于由神经网络参数化的具有大规模状态约束的OCP的基于模型的求解器。

3) We evaluate the proposed method extensively in both simulations and in a real-world road to verify the performance in terms of online computing efficiency, safety, and task adaptation, etc. The results show the potential of the method to be applied in real-world autonomous driving tasks.

3)我们在模拟和实际道路中广泛评估了所提出的方法，以验证其在在线计算效率、安全性和任务适应性等方面的性能。结果表明该方法在实际自动驾驶任务中的应用潜力。

## II. INTEGRATED DECISION AND CONTROL FRAMEWORK

## 二、集成决策与控制框架

In this section, we introduce the framework of integrated decision and control framework (IDC). As shown in Fig. 1 the framework consists of two layers: static path planning and dynamic optimal tracking.

在本节中，我们介绍集成决策与控制框架(IDC)。如图1所示，该框架由两层组成:静态路径规划和动态最优跟踪。

Different from existing schemes, the upper layer aims to generate multiple candidate paths only considering static information such as road structure, speed limit, traffic signs and lights. Note that these paths will not include time information. Each candidate path is attached with an expected velocity determined by rules from human experience.

与现有方案不同，上层旨在仅考虑道路结构、限速、交通标志和信号灯等静态信息来生成多条候选路径。请注意，这些路径不包含时间信息。每条候选路径都附有一个由人类经验规则确定的预期速度。

The lower layer further considers the candidate paths and the dynamic information such as surrounding vehicles, pedestrians and bicycles. For each candidate path, a constrained optimal control problem (OCP) is formulated and optimized to choose the optimal path and find the control command. The objective function is to minimize the tracking error within a finite horizon and the constraints characterize safety requirements. In each time step, the optimal path is chosen as the one with the lowest optimal cost function and thereafter tracked. The core of our method is to substitute all the expensive online optimizations with feed-forward of two neural networks (NNs) trained offline by reinforcement learning (RL). Specifically, we first formulate a complete RL problem considering all the candidate paths. And then we develop a model-based RL algorithm to solve this problem to obtain a policy NN called actor that is capable of tracking different shape of paths while maintaining the ability to avoid collisions. Meanwhile, a value NN called critic is learned to approximate the optimal cost of tracking different paths, for the purpose of online path selection.

下层进一步考虑候选路径以及周围车辆、行人和自行车等动态信息。对于每条候选路径，制定并优化一个约束最优控制问题(OCP)，以选择最优路径并找到控制命令。目标函数是在有限时域内最小化跟踪误差，约束条件表征安全要求。在每个时间步，选择最优成本函数最低的路径作为最优路径并进行跟踪。我们方法的核心是用两个通过强化学习(RL)离线训练的神经网络(NN)的前馈来替代所有昂贵的在线优化。具体来说，我们首先考虑所有候选路径制定一个完整的RL问题。然后我们开发一种基于模型的RL算法来解决这个问题，以获得一个称为智能体的策略NN，它能够跟踪不同形状的路径，同时保持避免碰撞的能力。同时，学习一个称为评论家的值NN来近似跟踪不同路径的最优成本，用于在线路径选择。

The advantages of the IDC framework are summarized in three points. First, it has high online computing efficiency. The upper layer can be quite efficient because it involves only static information. It even allows to embed key parameters of paths planned in advance into the electronic map and read from it directly. On the other hand, the lower layer utilizes two trained networks to compute optimal path and control command, which is also time-saving due to the fast propagation of NNs. Second, it can be easily transferred among different driving scenarios without a lot of human design. As the upper layer only uses the static information, the multiple paths can be easily generated by the road topology of various scenes such as intersections, roundabouts and ramps. Besides, the lower-layer always formulates a similar tracking problem with safety constraints no matter what the task is, which all can be solved by the developed model-based solver, saving time to design separate reward functions for different tasks. Third, the IDC framework are interpretable in the way that the learned value and policy function approximate the optimal value and the optimal action of the constrained OCPs. These optimal solutions can also be obtained by model predictive control (MPC) to verify the correctness of the trained NNs. Besides, the functionality partitioning in the IDC framework helps us to explain the results of path selection and tracking accordingly.

IDC框架的优点总结为三点。第一，它具有高在线计算效率。上层效率很高，因为它只涉及静态信息。它甚至允许将预先规划的路径的关键参数嵌入电子地图并直接从中读取。另一方面，下层利用两个训练好的网络来计算最优路径和控制命令，由于NN的快速传播，这也很节省时间。第二，它可以很容易地在不同驾驶场景之间转移，无需大量人工设计。由于上层仅使用静态信息，通过各种场景(如十字路口、环形交叉路口和匝道)的道路拓扑结构可以轻松生成多条路径。此外，无论任务是什么，下层总是制定一个带有安全约束的类似跟踪问题，所有这些都可以通过开发的基于模型的求解器来解决，从而节省为不同任务设计单独奖励函数的时间。第三，IDC框架是可解释的，因为学习到的值和策略函数近似约束OCP的最优值和最优动作。这些最优解也可以通过模型预测控制(MPC)获得，以验证训练好的NN的正确性。此外，IDC框架中的功能划分有助于我们相应地解释路径选择和跟踪的结果。

![bo_d7lhveilb0pc73f2i630_2_365_157_1070_655_0.jpg](images/bo_d7lhveilb0pc73f2i630_2_365_157_1070_655_0.jpg)

Fig. 1: Illustration of the integrated decision and control framework.

图1:集成决策与控制框架示意图。

## III. STATIC PATH PLANNING

## 三、静态路径规划

This module aims to generate multiple candidate paths for optimal tracking of the lower layer meanwhile maintaining high computational efficiency. For that purpose, the Cubic Bezier curve is adopted to obtain a continuous and smooth path. Given the road map, we choose four feature points to shape of Bezier curve and generate multiple paths with considerations of static traffic information such as road topology, traffic rules, etc., following the Algorithm 1 We summary two strategies to generate candidate paths. One is the pre-generating method, in which these paths are produced in advance and do not change with the riding of the ego vehicle, as illustrated in Fig. 1 The other one is the real-time generating method where the paths are planned in real-time and always start from the ego vehicle. Apparently, the former is simple and convenient to be directly embedded in the map, and has higher efficiency because it only needs to read the pre-stored paths every step. By contrast, the latter has higher flexibility and may conduct more complex driving behaviors, but its online computing efficiency will be a bit lower than the former. Both methods will be much more efficient than current existing methods as they simply generate regular candidate paths without considering collision avoidance with the dynamic obstacles. Note that the planned paths only serve as references for the lower layer, the actual travelled path can be largely different due to further considerations on dynamic obstacles.

该模块旨在生成多条候选路径，以便下层进行最优跟踪，同时保持高计算效率。为此，采用三次贝塞尔曲线来获得连续平滑的路径。给定道路地图，我们选择四个特征点来塑造贝塞尔曲线的形状，并考虑道路拓扑、交通规则等静态交通信息生成多条路径，遵循算法1。我们总结了两种生成候选路径的策略。一种是预生成方法，其中这些路径是预先生成的，不会随着自车的行驶而改变，如图1所示。另一种是实时生成方法，其中路径是实时规划的，并且总是从自车开始。显然，前者简单方便，可以直接嵌入地图，并且效率更高，因为它只需要在每一步读取预先存储的路径。相比之下，后者具有更高的灵活性，可能会进行更复杂的驾驶行为，但其在线计算效率会比前者略低。这两种方法都将比当前现有方法效率高得多，因为它们只是简单地生成规则的候选路径，而不考虑与动态障碍物的碰撞避免。请注意，规划的路径仅作为下层的参考，由于对动态障碍物的进一步考虑，实际行驶路径可能会有很大不同。

Algorithm 1 Static path planning

算法1 静态路径规划

---

Initialize: discrete point number \( M \)

for each path do

	Choose four feature points: start point \( \left( {{X}_{0},{Y}_{0}}\right) \) , control

	points \( \left( {{X}_{1},{Y}_{1}}\right) ,\left( {{X}_{2},{Y}_{2}}\right) \) , and end point \( \left( {{X}_{3},{Y}_{3}}\right) \)

	for \( t = 0 : 1/M : 1 \) do

		\( {p}_{\mathrm{x}}^{\text{ ref }}\left( t\right)  = {X}_{0}{\left( 1 - t\right) }^{3} + 3{X}_{1}t{\left( 1 - t\right) }^{2} + 3{X}_{2}{t}^{2}\left( {1 - t}\right)  + {X}_{3}{t}^{3} \)

		\( {p}_{\mathrm{v}}^{\text{ ref }}\left( t\right)  = {Y}_{0}{\left( 1 - t\right) }^{3} + 3{Y}_{1}t{\left( 1 - t\right) }^{2} + 3{Y}_{2}{t}^{2}\left( {1 - t}\right)  + {Y}_{3}{t}^{3} \)

		\( {\phi }^{\text{ ref }}\left( t\right)  = \arctan \left( \frac{Y\left( t\right)  - Y\left( {t - 1}\right) }{X\left( t\right)  - X\left( {t - 1}\right) }\right) \)

	end for

	Output \( \left\{  \left( {{p}_{\mathrm{x}}^{\text{ ref }},{p}_{\mathrm{y}}^{\text{ ref }},{\phi }^{\text{ ref }}}\right) \right\} \)

end for

---

As for the expected velocity, we heuristically assign different speed levels with respect to road regions, traffic signals as well as traffic rules such as speed limits or stop signs, which can be quickly designed according to human knowledge. An example is shown in Fig. 1 Similar to the candidate paths, the expected velocity provides a goal for the lower layer to track but not necessarily to follow strictly, so that the ego vehicle seeks to minimize the tracking error while satisfying safety constraints. Actually, it can be simply a fixed value, the lower layer will still always learn a driving policy to balance the safety requirements and tracking errors.

至于预期速度，我们根据道路区域、交通信号以及速度限制或停车标志等交通规则启发式地分配不同的速度级别，这可以根据人类知识快速设计。图1给出了一个示例。与候选路径类似，预期速度为下层提供了一个跟踪目标，但不一定严格遵循，以便自车在满足安全约束的同时寻求最小化跟踪误差。实际上，它可以只是一个固定值，下层仍然会始终学习一种驾驶策略来平衡安全要求和跟踪误差。

## IV. DYNAMIC OPTIMAL TRACKING

## 四、动态最优跟踪

## A. Problem formulation

## A. 问题表述

In each time step \( t \) , provided multiple candidate paths generated, the lower layer is designed to first select an optimal path \( {\tau }^{ * } \in  \Pi \) according to a certain criterion, where \( \Pi \) denotes a collection of \( N \) candidate paths. And then it obtains the control quantities \( {u}_{t} \) by optimizing a finite horizon constrained OCP, in which the objective is to minimize the tracking error as well as the control energy, and the constraints are to keep a safe distance from obstacles:

在每个时间步\( t \)，若生成了多条候选路径，下层首先根据某一准则选择一条最优路径\( {\tau }^{ * } \in  \Pi \)，其中\( \Pi \)表示\( N \)条候选路径的集合。然后通过优化一个有限时域约束最优控制问题(OCP)来获得控制量\( {u}_{t} \)，该OCP的目标是使跟踪误差和控制能量最小化，约束条件是与障碍物保持安全距离:

\[
\mathop{\min }\limits_{{{u}_{i \mid  t}, i = 0 : T - 1}}\;J = \mathop{\sum }\limits_{{i = 0}}^{{T - 1}}{\left( {x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}\right) }^{\top }Q\left( {{x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}}\right)  + {u}_{i \mid  t}^{\top }R{u}_{i \mid  t}
\]

\[
\text{ s.t. }\;{x}_{i + 1 \mid  t} = {F}_{\text{ ego }}\left( {{x}_{i \mid  t},{u}_{i \mid  t}}\right) \text{ , }
\]

\[
{x}_{i + 1 \mid  t}^{j} = {F}_{\text{ pred }}\left( {x}_{i \mid  t}^{j}\right) ,
\]

\[
{\left( {x}_{i \mid  t} - {x}_{i \mid  t}^{j}\right) }^{\top }M\left( {{x}_{i \mid  t} - {x}_{i \mid  t}^{j}}\right)  \geq  {D}_{\text{ veh }}^{\text{ safe }},
\]

\[
{\left( {x}_{i \mid  t} - {x}_{i \mid  t}^{\text{ road }}\right) }^{\top }M\left( {{x}_{i \mid  t} - {x}_{i \mid  t}^{\text{ road }}}\right)  \geq  {D}_{\text{ road }}^{\text{ safe }},
\]

\[
{x}_{i \mid  t} \leq  {L}_{\text{ stop }}\text{ , if light = red }
\]

\[
{x}_{0 \mid  t} = {x}_{t},{x}_{0 \mid  t}^{j} = {x}_{t}^{j},{u}_{0 \mid  t} = {u}_{t}
\]

\[
i = 0 : T - 1, j \in  I
\]

(1)

where \( T \) is the prediction horizon, \( {x}_{i \mid  t} \) and \( {u}_{i \mid  t} \) are the ego vehicle state and control in the virtual predictive time step \( i \) starting from the current time step \( t \) , where "virtual" means the time steps within the predictive horizon. \( {x}_{i \mid  t}^{\text{ ref }} \) and \( {x}_{i \mid  t}^{\text{ road }} \) are the closest point from \( {x}_{i \mid  t} \) on the selected reference \( {\tau }^{ * } \) and on the road edge, respectively. \( {x}_{i \mid  t}^{j} \) is the state of the \( j \) -th vehicle in the interested vehicle set \( I.Q, R, M \) are positive-definite weighting matrices. \( {F}_{\text{ ego }} \) represents the bicycle vehicle dynamics with linear tire model. \( {F}_{\text{ pred }} \) , on the other hand, is the surrounding vehicle prediction model. Besides, \( {D}_{\text{ veh }}^{\text{ safe }} \) and \( {D}_{\text{ road }}^{\text{ safe }} \) denote the safe distance from other vehicles and the road edge. \( {L}_{\text{ stop }} \) is the position of stop line. Note that in (1) the virtual states are all produced by the dynamics model and the prediction model except that the \( {x}_{0 \mid  t} \) is assigned with the current real state \( {x}_{t} \) . The variables and functions in (1) are further defined as:

其中\( T \)是预测时域，\( {x}_{i \mid  t} \)和\( {u}_{i \mid  t} \)分别是从当前时间步\( t \)开始的虚拟预测时间步\( i \)中的自车状态和控制量，这里的“虚拟”指预测时域内的时间步。\( {x}_{i \mid  t}^{\text{ ref }} \)和\( {x}_{i \mid  t}^{\text{ road }} \)分别是自车状态\( {x}_{i \mid  t} \)在所选参考路径\( {\tau }^{ * } \)上和道路边缘上的最近点。\( {x}_{i \mid  t}^{j} \)是感兴趣车辆集合中第\( j \)辆车的状态，\( I.Q, R, M \)是正定加权矩阵。\( {F}_{\text{ ego }} \)表示采用线性轮胎模型的自行车动力学。另一方面，\( {F}_{\text{ pred }} \)是周围车辆预测模型。此外，\( {D}_{\text{ veh }}^{\text{ safe }} \)和\( {D}_{\text{ road }}^{\text{ safe }} \)分别表示与其他车辆和道路边缘的安全距离。\( {L}_{\text{ stop }} \)是停车线的位置。注意，在式(1)中，除了\( {x}_{0 \mid  t} \)被赋予当前真实状态\( {x}_{t} \)外，虚拟状态均由动力学模型和预测模型产生。式(1)中的变量和函数进一步定义如下:

(2)

\[
{x}_{i \mid  t}^{\text{ ref }} = {\left\lbrack  \begin{matrix} {p}_{\mathrm{x}}^{\text{ ref }} \\  {p}_{\mathrm{y}}^{\text{ ref }} \\  {p}_{\mathrm{l}}^{\text{ ref }} \\  0 \\  {\phi }^{\text{ ref }} \\  0 \end{matrix}\right\rbrack  }_{i \mid  t}{x}_{i \mid  t}^{\text{ road }} = {\left\lbrack  \begin{matrix} {p}_{\mathrm{x}}^{\text{ road }} \\  {p}_{\mathrm{y}}^{\text{ road }} \\  0 \\  0 \\  0 \\  0 \end{matrix}\right\rbrack  }_{i \mid  t}{x}_{i \mid  t} = {\left\lbrack  \begin{matrix} {p}_{\mathrm{x}} \\  {p}_{\mathrm{y}} \\  {v}_{\text{ lon }} \\  {v}_{\text{ lat }} \\  \omega \\  \omega  \end{matrix}\right\rbrack  }_{i \mid  t}
\]

\[
{x}_{i \mid  t}^{j} = {\left\lbrack  \begin{matrix} {p}_{x}^{j} \\  {p}_{y}^{j} \\  {v}_{\text{ lon }}^{j} \\  0 \\  {\phi }^{j} \\  0 \end{matrix}\right\rbrack  }_{i \mid  t}{u}_{i \mid  t} = {\left\lbrack  \begin{matrix} \delta \\  a \end{matrix}\right\rbrack  }_{i \mid  t}M = \left\lbrack  \begin{array}{ll} 1 & 0 \\  0 & 1 \end{array}\right\rbrack
\]

\[
{F}_{\text{ ego }} = \left\lbrack  \begin{matrix} {p}_{\mathrm{x}} + {\Delta t}\left( {{v}_{\text{ lon }}\cos \phi  - {v}_{\text{ lat }}\sin \phi }\right) \\  {p}_{\mathrm{y}} + {\Delta t}\left( {{v}_{\text{ lon }}\sin \phi  + {v}_{\text{ lat }}\cos \phi }\right) \\  {v}_{\text{ lon }} + {\Delta t}\left( {a + {v}_{\text{ lat }}\omega }\right) \\  \frac{m{v}_{\text{ lon }}{v}_{\text{ lat }} + {\Delta t}\left\lbrack  {\left( {{L}_{f}{k}_{f} - {L}_{r}{k}_{r}}\right) \omega  - {k}_{f}\delta {v}_{\text{ lon }} - m{v}_{\text{ lon }}^{2}\omega }\right\rbrack  }{m{v}_{\text{ lon }} - {\Delta t}\left( {{k}_{f} + {k}_{r}}\right) } \\   + {\Delta t\omega } \\  \frac{-{I}_{z}\omega {v}_{\text{ lon }} - {\Delta t}\left\lbrack  {\left( {{L}_{f}{k}_{f} - {L}_{r}{k}_{r}}\right) {v}_{\mathrm{y}} - {L}_{f}{k}_{f}\delta {v}_{\text{ lon }}}\right\rbrack  }{{\Delta t}\left( {{L}_{f}^{2}{k}_{f} + {L}_{r}^{2}{k}_{r}}\right)  - {I}_{z}{v}_{\text{ lon }}} \end{matrix}\right\rbrack \tag{3}
\]

\[
{F}_{\text{ pred }} = \left\lbrack  \begin{matrix} {p}_{\mathrm{x}}^{j} + {\Delta t}\left( {{v}_{\text{ lon }}^{j}\cos {\phi }^{j} - {v}_{\text{ lat }}^{j}\sin {\phi }^{j}}\right) \\  {p}_{\mathrm{y}}^{j} + {\Delta t}\left( {{v}_{\text{ lon }}^{j}\sin {\phi }^{j} + {v}_{\text{ lat }}^{j}\cos {\phi }^{j}}\right) \\  {v}_{\text{ lon }}^{j} \\  0 \\  {\phi }^{j} + {\Delta t}{\omega }_{\text{ pred }}^{j} \\  0 \end{matrix}\right\rbrack \tag{4}
\]

where \( {p}_{\mathrm{x}},{p}_{\mathrm{y}} \) are the position coordinates, for ego and other vehicles, it is the position of their respective center of gravity (CG), \( {v}_{\text{ lon }},{v}_{\text{ lat }} \) are the longitudinal and lateral velocities, \( \phi \) is the heading angle, \( \omega \) is the yaw rate, \( \delta \) and \( a \) are the front wheel angle and the acceleration commands, respectively. The ego dynamics model is discretized by the first-order Euler method, which has been proved to be numerically stable at any low speed [31]. Other vehicle parameters are listed in Table 1 The vehicle prediction model is a simple deduction from the current state with constant speed and turning rate \( {w}_{\text{ pred }}^{j} \) , which depends on the driving scenarios and will be specified in the experiments.

其中\( {p}_{\mathrm{x}},{p}_{\mathrm{y}} \)是自车和其他车辆的位置坐标，即它们各自重心(CG)的位置，\( {v}_{\text{ lon }},{v}_{\text{ lat }} \)是纵向和横向速度，\( \phi \)是航向角，\( \omega \)是横摆率，\( \delta \)和\( a \)分别是前轮转角和加速度指令。自车动力学模型采用一阶欧拉方法离散化，已证明在任何低速情况下数值稳定[31]。其他车辆参数列于表1。车辆预测模型是根据当前状态以恒定速度和转弯率\( {w}_{\text{ pred }}^{j} \)进行的简单推导，它取决于驾驶场景，并将在实验中具体说明。

The optimal path is chosen as the one with the best tracking performance while satisfying the safety requirements, i.e.,

最优路径被选为在满足安全要求的同时具有最佳跟踪性能的路径，即

\[
{\tau }^{ * } = \arg \mathop{\min }\limits_{\tau }\left\{  {{J}_{\tau }^{ * } \mid  \tau  \in  \Pi }\right\} \tag{5}
\]

where \( {J}_{\tau }^{ * } \) is the optimal cost of the path \( \tau \) . This means that for each path candidate \( \tau  \in  \Pi \) we ought to construct such an OCP 1) and to optimize it to obtain its optimal value \( {J}_{\tau }^{ * } \) . Such a criterion of path selection is consistent with the objective of the optimal control problem (1) in the sense that it optimizes the problem with respect to paths within the candidate path set, compared to the original optimization with respect to control quantities.

其中\( {J}_{\tau }^{ * } \)是路径\( \tau \)的最优代价。这意味着对于每个候选路径\( \tau  \in  \Pi \)，我们都应构建这样一个OCP 1)并对其进行优化以获得其最优值\( {J}_{\tau }^{ * } \)。这种路径选择准则与最优控制问题(1)的目标一致，因为与原始针对控制量的优化相比，它是在候选路径集内针对路径对问题进行优化。

In such framework of selecting and tracking, the lower layer is able to well determine a control quantity that yields good driving efficiency and, meanwhile, the safety guarantee. But unfortunately, so far, the time complexity is extremely high for on-board vehicle computing devices, as in each time step we need to solve \( N \) constrained optimal control problem, while each of them owns up to hundreds of variables and thousands of constrains. Therefore, we employ RL to unload the online optimization burden. Specifically, we show that this framework naturally corresponds to the actor-critic architecture of RL, where the critic, with the function of judging state goodness, can be served as the path selector while the actor is in charge of action output and used for tracking. With a paradigm of offline training and online application, the computation burden in the lower layer can be almost nearly eliminated.

在这种选择和跟踪框架下，下层能够很好地确定一个能产生良好驾驶效率并同时保证安全的控制量。但不幸的是，到目前为止，对于车载计算设备而言时间复杂度极高，因为在每个时间步我们都需要求解\( N \)个约束最优控制问题，而每个问题都有多达数百个变量和数千个约束。因此，我们采用强化学习来减轻在线优化负担。具体而言，我们表明该框架自然对应于强化学习的演员 - 评论家架构，其中具有判断状态优劣功能的评论家可作为路径选择器，而演员负责动作输出并用于跟踪。通过离线训练和在线应用的范式，下层的计算负担几乎可以被消除。

## B. Offline training

## B. 离线训练

1) Complete RL problem formulation: We aim to solve the path selecting and path tracking problems 1) and 5) by RL. Notice that there are several significant differences between the OCP (1) and RL problems, originated from the online or offline optimizations. The OCP, which is designed for online optimization, seeks to find a single optimal control quantity of a single state given a specific path at time step \( t \) . RL problems, on the other hand, aim to solve a parameterized control policy called actor that maps from state space \( \mathcal{S} \) to control space \( \mathcal{A} \) as well as a parameterized value function called critic that evaluates the preference to a state, in an offline style. Therefore, in RL problems the variable to be optimized is no longer the control quantity but the parameters of the actor and critic that usually in the form of NN. In addition, the objective function and constraints of RL are not about a single state any more, but about a state distribution in the state space. The state \( s \in  \mathcal{S} \) is the input of the actor and critic, which is designed to contain necessary information to determine driving actions. Except for the information of the ego vehicle and surrounding vehicles, we also incorporate the path information as a part of states to get a policy that can handle tracking tasks for different paths. The resulting RL problem is shown as:

1) 完整的强化学习问题公式化:我们旨在通过强化学习解决路径选择和路径跟踪问题1) 和5)。注意，最优控制问题 (1) 和强化学习问题之间存在一些显著差异，这源于在线或离线优化。为在线优化设计的最优控制问题，在时间步\( t \)给定特定路径的情况下，寻求找到单个状态的单个最优控制量。另一方面，强化学习问题旨在以离线方式求解一个称为智能体的参数化控制策略，该策略从状态空间\( \mathcal{S} \)映射到控制空间\( \mathcal{A} \)，以及一个称为评论家的参数化价值函数，该函数评估对一个状态的偏好。因此，在强化学习问题中，要优化的变量不再是控制量，而是通常以神经网络形式存在的智能体和评论家的参数。此外，强化学习的目标函数和约束不再是关于单个状态，而是关于状态空间中的状态分布。状态\( s \in  \mathcal{S} \)是智能体和评论家的输入，其设计目的是包含确定驾驶动作所需的信息。除了自身车辆和周围车辆的信息外，我们还将路径信息作为状态的一部分纳入，以获得能够处理不同路径跟踪任务的策略。由此产生的强化学习问题如下所示:

\[
\mathop{\min }\limits_{\theta }\;{J}_{\text{ actor }} = {\mathbb{E}}_{{s}_{0 \mid  t}}\left\{  {\mathop{\sum }\limits_{{i = 0}}^{{T - 1}}l\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right) }\right\}
\]

\[
\text{ s.t. }{s}_{i + 1 \mid  t} = f\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right) \tag{6}
\]

\[
{g}_{e}\left( {s}_{i \mid  t}\right)  \geq  0, e \in  E
\]

\[
{s}_{0 \mid  t} = {s}_{t} \leftarrow  \left\{  {\tau ,{x}_{t},{x}_{t}^{j}, j \in  J}\right\}   \sim  d,
\]

\[
i = 0 : T - 1
\]

\( \mathop{\min }\limits_{w}{J}_{\text{ critic }} = {\mathbb{E}}_{{s}_{0 \mid  t}}\left\{  {\left( \mathop{\sum }\limits_{{i = 0}}^{{T - 1}}l\left( {s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) \right)  - {V}_{w}\left( {s}_{0 \mid  t}\right) \right) }^{2}\right\} \)

\[
\text{ s.t. }{s}_{i + 1 \mid  t} = f\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right)
\]

\[
{s}_{0 \mid  t} = {s}_{t} \sim  d,
\]

\[
i = 0 : T - 1
\]

(7)

where \( {s}_{t} \leftarrow  \left\{  {\tau ,{x}_{t},{x}_{t}^{j}, j \in  J}\right\} \) denotes the state is constructed using the information of reference path, ego vehicle state and surrounding vehicle states. \( l\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right)  \mathrel{\text{ := }} \; {\left( {x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}\right) }^{\top }Q\left( {{x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}}\right)  + {\pi }_{\theta }^{\top }\left( {s}_{i \mid  t}\right) R{\pi }_{\theta }\left( {s}_{i \mid  t}\right) .f \) denotes the system model, which is an aggregation of the \( {F}_{\text{ ego }} \) and \( {F}_{\text{ pred. }}{g}_{e}\left( {s}_{i \mid  t}\right) , e \in  E \) denotes all the constraints about the state \( {s}_{i \mid  t} \) , including that with other vehicles, road, and traffic rules. \( d \) denotes the state distributions sampled from the environment. \( {\pi }_{\theta } : \mathcal{S} \rightarrow  \mathcal{A} \) and \( {V}_{w} : \mathcal{S} \rightarrow  \mathbb{R} \) are actor and critic, parameterized by \( \theta \) and \( w \) that are generally in form of NNs, respectively. From the results of [32], given over-parameterized NNs, the optimal policy \( {\pi }_{{\theta }_{ * }} \) of (6) maps to an optimal action of the original OCP [1] with arbitrary initial state \( {s}_{t} \) . Consequently, the optimal value \( {J}^{ * } \) from [1], of course, would be equal to the one mapped by the optimal value function \( {V}_{{w}_{ * }} \) of [7] from \( {s}_{t} \) , i.e.,

其中\( {s}_{t} \leftarrow  \left\{  {\tau ,{x}_{t},{x}_{t}^{j}, j \in  J}\right\} \)表示使用参考路径、自身车辆状态和周围车辆状态的信息构建的状态。\( l\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right)  \mathrel{\text{ := }} \; {\left( {x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}\right) }^{\top }Q\left( {{x}_{i \mid  t}^{\text{ ref }} - {x}_{i \mid  t}}\right)  + {\pi }_{\theta }^{\top }\left( {s}_{i \mid  t}\right) R{\pi }_{\theta }\left( {s}_{i \mid  t}\right) .f \)表示系统模型，它是\( {F}_{\text{ ego }} \)的聚合，\( {F}_{\text{ pred. }}{g}_{e}\left( {s}_{i \mid  t}\right) , e \in  E \)表示关于状态\( {s}_{i \mid  t} \)的所有约束，包括与其他车辆、道路和交通规则相关的约束。\( d \)表示从环境中采样的状态分布。\( {\pi }_{\theta } : \mathcal{S} \rightarrow  \mathcal{A} \)和\( {V}_{w} : \mathcal{S} \rightarrow  \mathbb{R} \)分别是由通常为神经网络形式的\( \theta \)和\( w \)参数化的智能体和评论家。根据[32]的结果，给定过参数化的神经网络，(6) 的最优策略\( {\pi }_{{\theta }_{ * }} \)映射到具有任意初始状态\( {s}_{t} \)的原始最优控制问题 [1] 的最优动作。因此，[1] 中的最优值\( {J}^{ * } \)当然将等于由[7]的最优值函数\( {V}_{{w}_{ * }} \)从\( {s}_{t} \)映射得到的值，即

(8)

\[
{u}_{t}^{ * } = {\pi }_{{\theta }_{ * }}\left( {s}_{t}\right) ,\forall {s}_{t} \in  \mathcal{S}
\]

\[
{J}^{ * } = {V}_{{w}_{ * }}\left( {s}_{t}\right) ,\forall {s}_{t} \in  \mathcal{S}
\]

In other words, the optimal policy and value function can output the optimal control and value under arbitrary states, i.e., arbitrary combinations of paths, ego state and surrounding vehicle states.

换句话说，最优策略和价值函数可以在任意状态下输出最优控制和值，即路径、自身状态和周围车辆状态的任意组合。

2) Solver - GEP: To solve the converted RL problem, we adopt the policy iteration framework, wherein two procedures, namely policy evaluation and policy improvement, are alternatively performed to update the critic and actor. Since the critic update is an unconstrained problem that can be optimized by ordinary gradient descent methods, we mainly focus on the actor update which is quite challenging because of its large-scale parameter space, nonlinear property and infinite number of state constraints. To tackle this, we propose a model-based RL algorithm called generalized exterior point method (GEP) adapted from the one in the optimization field. It first transforms the constrained problem 6 into an unconstrained one by the exterior penalty function, shown as:

2) 求解器 - 广义外点法:为了解决转换后的强化学习问题，我们采用策略迭代框架，其中交替执行策略评估和策略改进两个过程来更新评论家与智能体。由于评论家更新是一个无约束问题，可以通过普通梯度下降方法进行优化，我们主要关注智能体更新，由于其大规模参数空间、非线性特性和无限数量的状态约束，这极具挑战性。为了解决这个问题，我们提出一种基于模型的强化学习算法，称为广义外点法 (GEP)，它改编自优化领域的一种方法。它首先通过外部惩罚函数将约束问题6转换为无约束问题，如下所示:

\[
\mathop{\min }\limits_{\theta }\;{J}_{p} = {J}_{\text{ actor }} + \rho {J}_{\text{ penalty }}
\]

\[
= {\mathbb{E}}_{{s}_{0 \mid  t}}\left\{  {\mathop{\sum }\limits_{{i = 0}}^{{T - 1}}l\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right) }\right\}   + \rho {\mathbb{E}}_{{s}_{0 \mid  t}}\left\{  {\mathop{\sum }\limits_{{i = 0}}^{{T - 1}}{\varphi }_{i}\left( \theta \right) }\right\}
\]

\[
\text{ s.t. }\;{s}_{i + 1 \mid  t} = f\left( {{s}_{i \mid  t},{\pi }_{\theta }\left( {s}_{i \mid  t}\right) }\right)
\]

\[
{\varphi }_{i}\left( \theta \right)  = \mathop{\sum }\limits_{{e \in  E}}{\left\lbrack  \max \left\{  0, - {g}_{e}\left( {s}_{i \mid  t}\right) \right\}  \right\rbrack  }^{2}
\]

\[
{s}_{0 \mid  t} = {s}_{t} \sim  d,
\]

\[
i = 0 : T - 1
\]

(9)

where \( \varphi \) is the penalty function, \( \rho \) is the penalty factor. After that, we alternatively optimize the policy parameters by performing \( m \) iterations of gradient descent and increase the penalty factor by multiplying a scalar \( c > 1 \) . We call the former the optimizing procedure and the latter the amplifying procedure. Different from exterior point method, the optimizing procedure of GEP does not necessarily find the optimal solution of the unconstrained problem [9], but we will prove that GEP still converges to the optimal policy under certain conditions. It can be seen that GEP is simple to implement and is powerful to deal with large-scale parameter space facilitated by the gradient descent technique. Besides, from the form of 9), numerous state constraints can be handled naturally by regarding the constraint violation as a term of utility function multiplied by \( \rho \) . The training pipeline is shown in Algorithm 2

其中\( \varphi \)为惩罚函数，\( \rho \)为惩罚因子。在此之后，我们交替地通过执行\( m \)次梯度下降迭代来优化策略参数，并通过将标量\( c > 1 \)相乘来增加惩罚因子。我们将前者称为优化过程，后者称为放大过程。与外点法不同，GEP的优化过程不一定能找到无约束问题的最优解[9]，但我们将证明GEP在某些条件下仍能收敛到最优策略。可以看出，GEP易于实现，并且借助梯度下降技术能够有效地处理大规模参数空间。此外，从式(9)的形式来看，通过将违反约束的情况视为效用函数乘以\( \rho \)的一项，可以自然地处理众多状态约束。训练流程如算法2所示

Next, we present the convergence proof of GEP. We say a "round" completes when an optimizing procedure is finished. Since the optimizing procedure only improves [9] a fixed number of times to obtain a fair solution but does not necessarily find the optimal one, we first give an assumption about how well the solution is.

接下来，我们给出GEP的收敛性证明。当一个优化过程完成时，我们称完成了一轮。由于优化过程仅改进[9]固定次数以获得一个合理的解，但不一定能找到最优解，我们首先对解的质量给出一个假设。

Assumption 1. After the round \( k \) completes, we have the penalty factor \( {\rho }_{k} \) and an optimized policy parameter \( {\theta }_{k} \) . We assume that \( {\theta }_{k} \) satisfies

假设1。在第\( k \)轮完成后，我们有惩罚因子\( {\rho }_{k} \)和一个优化后的策略参数\( {\theta }_{k} \)。我们假设\( {\theta }_{k} \)满足

\[
{J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  \leq  \mathop{\min }\limits_{\theta }{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  + {\Delta }_{k}, k = 1,2,\ldots \tag{10}
\]

where \( {\Delta }_{k} \geq  0, k \geq  1 \) is a positive non-increasing sequence that has finite series, i.e., \( {\Delta }_{k} \geq  {\Delta }_{k + 1},\mathop{\sum }\limits_{{i = 0}}^{\infty }{\Delta }_{i} < \infty \) .

其中\( {\Delta }_{k} \geq  0, k \geq  1 \)是一个正的非递增序列，其级数有限，即\( {\Delta }_{k} \geq  {\Delta }_{k + 1},\mathop{\sum }\limits_{{i = 0}}^{\infty }{\Delta }_{i} < \infty \)。

Algorithm 2 Dynamic optimal tracking - Offline training

算法2 动态最优跟踪 - 离线训练

---

Initialize: critic network \( {V}_{w} \) and actor network \( {\pi }_{\theta } \) with

random paramaters \( w,\theta \) , buffer \( \mathcal{B} \leftarrow  \varnothing \) , learning rates

\( {\beta }_{w},{\beta }_{\theta } \) , penalty factor \( \rho  = 1 \) , penalty amplifier \( c \) , update

interval \( m \)

for each iteration \( i \) do

	// Sampling (from environment)

	Randomly select a path \( \tau  \in  \Pi \) , initialize ego state \( {x}_{t} \) and

	vehicle states \( {x}_{t}^{j}, j \in  I \)

	for each environment step do

		\( {s}_{t} \leftarrow  \left\{  {\tau ,{x}_{t},{x}_{t}^{j}, j \in  I}\right\} \)

		\( \mathcal{B} \cup  \left\{  {s}_{t}\right\} \)

		\( {u}_{t} = {\pi }_{\theta }\left( {s}_{t}\right) \)

		Apply \( {u}_{t} \) to observe \( {x}_{t + 1} \) and \( {x}_{t + 1}^{j}, j \in  I \)

	end for

	// Optimizing (GEP)

	Fetch a batch of states from \( \mathcal{B} \) , compute \( {J}_{\text{ critic }} \) and \( {J}_{p} \)

	by \( f \) and \( {\pi }_{\theta } \)

	PEV: \( w \leftarrow  w - {\beta }_{w}{\nabla }_{w}{J}_{\text{ critic }} \)

	PIM: if \( i{\;\operatorname{mod}\;m} : \rho  \leftarrow  {c\rho };\theta  \leftarrow  \theta  - {\beta }_{\theta }{\nabla }_{\theta }{J}_{p} \)

end for

---

The Assumption 1 describes that with the convergence of NNs, the gap between the solution of the optimizing procedure and the optimal one is gradually eliminated, i.e., \( \mathop{\lim }\limits_{{k \rightarrow  \infty }}{\Delta }_{k} = 0 \) , as indicated by the finite series. About the gap, we have the following Lemma.

假设1描述了随着神经网络的收敛，优化过程的解与最优解之间的差距逐渐消除，即\( \mathop{\lim }\limits_{{k \rightarrow  \infty }}{\Delta }_{k} = 0 \)，如有限级数所示。关于这个差距，我们有以下引理。

Lemma 1. There exists a positive non-increasing sequence \( {\delta }_{k}, k \geq  1 \) that satisfies

引理1。存在一个正的非递增序列\( {\delta }_{k}, k \geq  1 \)满足

\[
{\Delta }_{k} = {\delta }_{k} - {\delta }_{k + 1}
\]

\[
{\delta }_{k} \geq  {\delta }_{k + 1},\mathop{\lim }\limits_{{k \rightarrow  \infty }}{\delta }_{k} = 0 \tag{11}
\]

Proof. We can construct such a sequence by setting

证明。我们可以通过设置来构造这样一个序列

\[
{\delta }_{1} = \mathop{\sum }\limits_{{i = 1}}^{\infty }{\Delta }_{i},{\delta }_{k + 1} = {\delta }_{k} - {\Delta }_{k}, k \geq  1 \tag{12}
\]

Then \( {\delta }_{1} < 0 \) holds by Assumption 1 and the convergence of \( {\delta }_{k} \) naturally holds by

那么由假设1可得\( {\delta }_{1} < 0 \)成立，并且由可得\( {\delta }_{k} \)自然成立

\[
\mathop{\lim }\limits_{{k \rightarrow  \infty }}{\delta }_{k} = \mathop{\lim }\limits_{{k \rightarrow  \infty }}{\delta }_{1} - \mathop{\sum }\limits_{{i = 1}}^{{k - 1}}{\Delta }_{i} = {\delta }_{1} - \mathop{\lim }\limits_{{k \rightarrow  \infty }}\mathop{\sum }\limits_{{i = 1}}^{{k - 1}}{\Delta }_{i} = 0 \tag{13}
\]

Next, we first prove the following two Lemmas about the unconstrained objective.

接下来，我们首先证明关于无约束目标的以下两个引理。

Lemma 2. For the solution sequence generated after each round \( \left\{  {\theta }_{k}\right\} \) , we have

引理2。对于每一轮\( \left\{  {\theta }_{k}\right\} \)之后生成的解序列，我们有

\[
{J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k + 1}}\right)  - {\delta }_{k + 1} \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\delta }_{k} \tag{14}
\]

Proof. By \( {J}_{p}\left( {\theta ,\rho }\right)  = {J}_{\text{ actor }}\left( \theta \right)  + \rho {J}_{\text{ penalty }}\left( \theta \right) \) and \( {\rho }_{k + 1} > {\rho }_{k} \) ,

证明。由\( {J}_{p}\left( {\theta ,\rho }\right)  = {J}_{\text{ actor }}\left( \theta \right)  + \rho {J}_{\text{ penalty }}\left( \theta \right) \)和\( {\rho }_{k + 1} > {\rho }_{k} \)，

\[
{J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k + 1}}\right)  = {J}_{\text{ actor }}\left( {\theta }_{k + 1}\right)  + {\rho }_{k + 1}{J}_{\text{ penalty }}\left( {\theta }_{k + 1}\right)
\]

\[
\geq  {J}_{\text{ actor }}\left( {\theta }_{k + 1}\right)  + {\rho }_{k}{J}_{\text{ penalty }}\left( {\theta }_{k + 1}\right) \tag{15}
\]

\[
= {J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k}}\right)
\]

Then by the Assumption 1 for \( \forall \theta ,{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  \geq \; \mathop{\min }\limits_{\theta }{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k} \) , thus we have \( {J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k} \) , therefore

然后根据假设1对于\( \forall \theta ,{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  \geq \; \mathop{\min }\limits_{\theta }{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k} \)，因此我们有\( {J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k} \)，所以

\[
{J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k + 1}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k} \tag{16}
\]

\[
{J}_{p}\left( {{\theta }_{k + 1},{\rho }_{k + 1}}\right)  - {\delta }_{k + 1} \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\delta }_{k}
\]

Lemma 3. Suppose \( {\theta }_{ * } = \arg \mathop{\min }\limits_{\theta }{J}_{\text{ actor }}\left( \theta \right) \) , then for \( \forall k \geq  1 \) ,

引理3。假设\( {\theta }_{ * } = \arg \mathop{\min }\limits_{\theta }{J}_{\text{ actor }}\left( \theta \right) \)，那么对于\( \forall k \geq  1 \)，

\[
{J}_{\text{ actor }}\left( {\theta }_{ * }\right)  - {\delta }_{k + 1} \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\delta }_{k} \geq  {J}_{\text{ actor }}\left( {\theta }_{k}\right)  - {\delta }_{k} \tag{17}
\]

Proof. Because \( {\theta }_{ * } \) is the optimal solution of the problem 6 , it has \( {J}_{\text{ penalty }}\left( {\theta }_{ * }\right)  = 0 \) . Then from the Assumption 1 the first inequality is obtained,

证明。因为\( {\theta }_{ * } \)是问题6的最优解，所以有\( {J}_{\text{ penalty }}\left( {\theta }_{ * }\right)  = 0 \)。然后根据假设1得到第一个不等式，

\[
{J}_{\text{ actor }}\left( {\theta }_{ * }\right)  = {J}_{p}\left( {{\theta }_{ * },{\rho }_{k}}\right)  \geq  \mathop{\min }\limits_{\theta }{J}_{p}\left( {\theta ,{\rho }_{k}}\right)  \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\Delta }_{k}
\]

\[
{J}_{\text{ actor }}\left( {\theta }_{ * }\right)  - {\delta }_{k + 1} \geq  {J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  - {\delta }_{k}
\]

(18)

The second inequality is got by \( {\rho }_{k}{J}_{\text{ penalty }}\left( {\theta }_{k}\right)  \geq  0 \)

第二个不等式由\( {\rho }_{k}{J}_{\text{ penalty }}\left( {\theta }_{k}\right)  \geq  0 \)得出

\[
{J}_{p}\left( {{\theta }_{k},{\rho }_{k}}\right)  = {J}_{\text{ actor }}\left( {\theta }_{k}\right)  + {\rho }_{k}{J}_{\text{ penalty }}\left( {\theta }_{k}\right)  \geq  {J}_{\text{ actor }}\left( {\theta }_{k}\right) \tag{19}
\]

The convergence can be revealed by Theorem 1

收敛性可由定理1揭示

Theorem 1. Assume that \( {J}_{\text{ actor }} \) and \( {J}_{\text{ penalty }} \) are continuous functions defined on the parameter space. Suppose \( \left\{  {\theta }_{k}\right\} \) is the solution sequence generated after each round. The limit of any of its convergent subsequence is the optimal solution.

定理1。假设\( {J}_{\text{ actor }} \)和\( {J}_{\text{ penalty }} \)是定义在参数空间上的连续函数。设\( \left\{  {\theta }_{k}\right\} \)是每一轮后生成的解序列。其任何收敛子序列的极限都是最优解。

Proof. Suppose \( \left\{  {\theta }_{{k}_{j}}\right\} \) is an arbitrary convergent subsequence of \( \left\{  {\theta }_{k}\right\} \) with the limit \( \overline{\theta } \) . By the continuity of \( {J}_{\text{ actor }},\mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}{J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  = {J}_{\text{ actor }}\left( \overline{\theta }\right) \) . Define \( {J}_{\text{ actor }}^{ * } = \; \mathop{\min }\limits_{\theta }{J}_{\text{ actor }}\left( \theta \right) \) as the optimal value of the problem 6 . From Lemma 2 and Lemma 3, we can see that \( \left\{  {{J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  - {\delta }_{{k}_{j}}}\right\} \) is a non-increasing sequence with upper limit \( {J}_{\text{ actor }}^{ * } \) , therefore

证明。假设\( \left\{  {\theta }_{{k}_{j}}\right\} \)是\( \left\{  {\theta }_{k}\right\} \)的任意一个收敛子序列，极限为\( \overline{\theta } \)。由\( {J}_{\text{ actor }},\mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}{J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  = {J}_{\text{ actor }}\left( \overline{\theta }\right) \)的连续性。定义\( {J}_{\text{ actor }}^{ * } = \; \mathop{\min }\limits_{\theta }{J}_{\text{ actor }}\left( \theta \right) \)为问题6的最优值。由引理2和引理3，我们可以看到\( \left\{  {{J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  - {\delta }_{{k}_{j}}}\right\} \)是一个上界为\( {J}_{\text{ actor }}^{ * } \)的非增序列，因此

\[
\mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}\left( {{J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  - {\delta }_{{k}_{j}}}\right)  = \mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}{J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  = {J}_{p}^{ * } \leq  {J}_{\text{ actor }}^{ * }
\]

(20)

Then because \( {J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  = {J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  + {\rho }_{{k}_{j}}{J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right) \) ,

那么因为\( {J}_{p}\left( {{\theta }_{{k}_{j}},{\rho }_{{k}_{j}}}\right)  = {J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  + {\rho }_{{k}_{j}}{J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right) \)

\[
\mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}{\rho }_{{k}_{j}}{J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right)  = {J}_{p}^{ * } - {J}_{\text{ actor }}\left( \overline{\theta }\right) \tag{21}
\]

by \( {J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right)  \geq  0,{\rho }_{{k}_{j}} \rightarrow  \infty \) and the continuity of \( {J}_{\text{ penalty }} \) , we have

由\( {J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right)  \geq  0,{\rho }_{{k}_{j}} \rightarrow  \infty \)以及\( {J}_{\text{ penalty }} \)的连续性，我们有

\[
\mathop{\lim }\limits_{{{k}_{j} \rightarrow  \infty }}{J}_{\text{ penalty }}\left( {\theta }_{{k}_{j}}\right)  = {J}_{\text{ penalty }}\left( \overline{\theta }\right)  = 0 \tag{22}
\]

which indicates that the limit \( \overline{\theta } \) is a feasible solution. Furthermore, by Lemma 3. \( {J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  \leq  {J}_{\text{ actor }}^{ * } \) , together with the continuity of \( {J}_{\text{ actor }} \) , we have

这表明极限\( \overline{\theta } \)是一个可行解。此外，由引理3。\( {J}_{\text{ actor }}\left( {\theta }_{{k}_{j}}\right)  \leq  {J}_{\text{ actor }}^{ * } \)，连同\( {J}_{\text{ actor }} \)的连续性，我们有

\[
{J}_{\text{ actor }}\left( \overline{\theta }\right)  \leq  {J}_{\text{ actor }}^{ * } \Rightarrow  {J}_{\text{ actor }}\left( \overline{\theta }\right)  = {J}_{\text{ actor }}^{ * } \tag{23}
\]

where the equation holds by the definition of \( {J}_{\text{ actor }}^{ * } \) . Equation 23 indicates the optimality of the limit \( \overline{\theta } \) .

其中等式由\( {J}_{\text{ actor }}^{ * } \)的定义成立。方程23表明极限\( \overline{\theta } \)的最优性。

## C. Online application

## C. 在线应用

Ideally, we expect to get an optimal policy which is able to output the optimal action within the safety action space in any given point in the state space. Unfortunately, it is impossible to acquire such a policy in both theory and practice. First, the converted RL problem 6 enforces constraint on every single state point, leading to an infinite number of constraints in the continuous state space. But nevertheless when we solve the equivalent unconstrained problem 9 , we approximate the expectation by an average of samples, that means we only consider finite constraints of the set of sample that can vary in different iterations. That is why there is no strict safety guarantee of the policy but only an approximately safe performance. Second, the condition of (8) that the approximation function has infinite fitting power cannot be established in practical, resulting in a suboptimal solution without safety guarantee. To ensure the safety performance, we adopt a multi-step safety shield after the output of the policy.

理想情况下，我们期望得到一个最优策略，该策略能够在状态空间中的任何给定时刻在安全动作空间内输出最优动作。不幸的是，在理论和实践中都不可能获得这样的策略。首先，转换后的强化学习问题6对每个单独的状态点都施加了约束，导致在连续状态空间中有无限数量的约束。但尽管如此，当我们求解等效的无约束问题9时，我们通过样本平均值来近似期望，这意味着我们只考虑样本集的有限约束，这些样本在不同迭代中可能会有所不同。这就是为什么策略没有严格的安全保证，而只有近似安全的性能。其次，(8)中逼近函数具有无限拟合能力的条件在实际中无法成立，导致得到一个没有安全保证的次优解。为了确保安全性能，我们在策略输出后采用多步安全防护。

1) Multi-step safety shield: The safety shield aims to find the nearest actions in the safe action space in state \( {s}_{t} \) , which is formulated as a quadratic programming problem:

1) 多步安全防护:安全防护旨在在状态\( {s}_{t} \)中找到安全动作空间中最近的动作，这被表述为一个二次规划问题:

\[
{u}_{t}^{\text{ safe }} = \left\{  \begin{array}{l} {u}_{t}^{ * },\;\text{ if }\;{u}_{t}^{ * } \in  {\mathcal{U}}_{\text{ safe }}\left( {s}_{t}\right) \\  \arg \mathop{\min }\limits_{{u \in  {\mathcal{U}}_{\text{ safe }}\left( {s}_{t}\right) }}{\begin{Vmatrix}u - {u}_{t}^{ * }\end{Vmatrix}}_{2}^{2},\;\text{ else } \end{array}\right. \tag{24}
\]

where \( {u}_{t}^{ * } \) is the policy output. Rather than designing \( {\mathcal{U}}_{\text{ safe }}\left( {s}_{t}\right) \) to guarantee the safety of only the next state, we design it to guarantee that the next \( {n}_{ss} \) prediction states are safe, i.e., collision-free with the surrounding vehicles and road edges. Formally,

其中\( {u}_{t}^{ * } \)为策略输出。我们设计\( {\mathcal{U}}_{\text{ safe }}\left( {s}_{t}\right) \)的目的不是仅保证下一状态的安全性，而是保证接下来的\( {n}_{ss} \)个预测状态是安全的，即与周围车辆和道路边缘无碰撞。形式上，

\[
{\mathcal{U}}_{\text{ safe }}\left( {s}_{t}\right)  = \left\{  {{u}_{t} \mid  {g}_{e}\left( {s}_{i \mid  t}\right)  \geq  0, i = 1,\ldots ,{n}_{ss}, e \in  E}\right\} \tag{25}
\]

2) Algorithm for online application: Given the trained policy and value functions, in online application, we simply construct a set of states for different paths, then pass them to the trained value function to select the one with the lowest value, which is next passed to the trained policy to get the optimal control, as summarized in Algorithm 3

2) 在线应用算法:给定训练好的策略和价值函数，在在线应用中，我们简单地为不同路径构建一组状态，然后将它们传递给训练好的价值函数以选择值最低的状态，接着将该状态传递给训练好的策略以获得最优控制，如算法3总结所示

Algorithm 3 Dynamic optimal tracking - Online application

算法3 动态最优跟踪 - 在线应用

---

Initialize: Path set \( \Pi \) from upper layer, trained critic net-

work \( {V}_{{w}^{ * }} \) and actor network \( {\pi }_{{\theta }^{ * }},\lambda \) , ego state \( {x}_{t} \) and vehicle

states \( {x}_{t}^{j}, j \in  I \)

for each environment step do

	// Selecting

	for each \( \tau  \in  \Pi \) do

		\( {s}_{t,\tau } \leftarrow  \left\{  {\tau ,{x}_{t},{x}_{t}^{j}, j \in  I}\right\} \)

		\( {V}_{\tau }^{ * } = {V}_{{w}^{ * }}\left( {s}_{t,\tau }\right) \)

	end for

	\( {\tau }^{ * } = \arg \mathop{\min }\limits_{\tau }\left\{  {{V}_{\tau }^{ * } \mid  \tau  \in  \Pi }\right\} \)

	// Tracking

	\( {s}_{t} \leftarrow  \left\{  {{\tau }^{ * },{x}_{t},{x}_{t}^{j}, j \in  I}\right\} \)

	\( {u}_{t}^{ * } = {\pi }_{{\theta }^{ * }}\left( {s}_{t}\right) \)

	Calculate \( {u}_{\text{ safe }}^{ * } \) by 24

	Apply \( {u}_{\text{ safe }}^{ * } \) to observe \( {x}_{t + 1} \) and \( {x}_{t + 1}^{j}, j \in  I \)

end for

---

## V. SIMULATION VERIFICATION

## V. 仿真验证

## A. Scenario and task descriptions

## A. 场景和任务描述

We first carried out our experiments on a regular signalized four-way intersection built in the simulation, where the roads in different directions are all the six-lane dual carriageway, as shown in Fig. 2 The junction is a square with a side length of \( {50}\mathrm{\;m} \) . Each entrance of the intersection has three lanes, each with a width of \( {3.75}\mathrm{\;m} \) , for turning left, going straight and turning right, respectively. With the help of SUMO [33], we generate a dense traffic flow of 800 vehicles per hour on each lane. These vehicles are controlled by the car-following and lane-changing models in the SUMO, producing a variety of traffic behaviors. Moreover, a two-phase traffic signal is included to control the traffic flow of turning left and going straight. We verify our algorithm in three tasks: turn left, go straight and turn right. In each task, the ego vehicle is initialized randomly from the south entrance and is expected to drive safely and efficiently to pass the intersection.

我们首先在模拟中构建的常规信号控制四向交叉路口进行实验，不同方向的道路均为六车道双车道，如图2所示。交叉路口是一个边长为\( {50}\mathrm{\;m} \)的正方形。交叉路口的每个入口有三条车道，每条车道宽度为\( {3.75}\mathrm{\;m} \)，分别用于左转、直行和右转。借助SUMO [33]，我们在每条车道上生成每小时800辆车的密集交通流。这些车辆由SUMO中的跟车和换道模型控制，产生各种交通行为。此外，还包括一个两相位交通信号来控制左转和直行的交通流。我们在三个任务中验证我们的算法:左转、直行和右转。在每个任务中，自车从南入口随机初始化，期望安全高效地驶过交叉路口。

![bo_d7lhveilb0pc73f2i630_6_982_837_610_546_0.jpg](images/bo_d7lhveilb0pc73f2i630_6_982_837_610_546_0.jpg)

Fig. 2: The scenario used for experiment verification.

图2:用于实验验证的场景。

## B. Implementation of our algorithm

## B. 我们算法的实现

1) Path planning: In this paper, we adopt the static path planning method to generate multiple candidate paths. Specially, in our scenario, each task is assigned three paths according to the lane number of exits. The paths are generated by the cubic Bezier curve featured by four key points. The expected velocity is chosen as a fixed value for simplicity.

1) 路径规划:在本文中，我们采用静态路径规划方法生成多条候选路径。具体而言，在我们的场景中，根据出口车道数为每个任务分配三条路径。路径由具有四个关键点的三次贝塞尔曲线生成。为简单起见，期望速度选为固定值。

2) State and utility function: As mentioned in section IV-B1 the state should be designed to include information of the ego vehicle, the surrounding vehicles and the reference path, i.e.,

2) 状态和效用函数:如第四节B1中所述，状态应设计为包含自车、周围车辆和参考路径的信息，即，

\[
{s}_{t} = \left\lbrack  {{s}_{t}^{\text{ ego }},{s}_{t}^{\text{ other }},{s}_{t}^{\text{ ref }}}\right\rbrack \tag{26}
\]

where \( {s}_{t}^{\text{ ego }} = {x}_{t} \) is the ego dynamics defined in section IV-A, \( {s}_{t}^{\text{ other }} \) is the concatenation of the interested surrounding vehicles. Take the turn left task as an example, it is defined as:

其中\( {s}_{t}^{\text{ ego }} = {x}_{t} \)是第四节A中定义的自车动力学，\( {s}_{t}^{\text{ other }} \)是感兴趣的周围车辆的串联。以左转任务为例，其定义为:

\[
{s}_{t}^{\text{ other }} = {\left\lbrack  {p}_{x}^{j},{p}_{y}^{j},{\phi }^{j},{v}_{\text{ lon }}^{j}\right\rbrack  }_{t, j \in  {I}_{\text{ left }}}
\]

\[
{I}_{\text{ left }} = \left\lbrack  {\mathrm{{SW}}1,\mathrm{{SW}}2,\mathrm{{SN}}1,\mathrm{{SN}}2,\mathrm{{NS}}1,\mathrm{{NS}}2,\mathrm{{NW}}1,\mathrm{{NW}}2}\right\rbrack
\]

(27)where \( {I}_{\text{ left }} \) is an ordered list of vehicles that have potential conflicts with the ego. They are encoded by their respect route start and end, as well as the order on that. Correspondingly, one can define the go straight and turn right task in a similar way. The information of the reference \( {s}_{t}^{\text{ ref }} \) , however, is designed in an implicit way by the tracking errors with respect to the position, the heading angle, and the velocity:

其中\( {I}_{\text{ left }} \)是与自车有潜在冲突的车辆的有序列表。它们通过各自的路线起点和终点以及在该路线上的顺序进行编码。相应地，可以以类似的方式定义直行和右转任务。然而，参考\( {s}_{t}^{\text{ ref }} \)的信息是通过相对于位置、航向角和速度的跟踪误差以隐式方式设计的:

\[
{s}_{t}^{\text{ ref }} = {\left\lbrack  {\delta }_{p},{\delta }_{\phi },{\delta }_{v}\right\rbrack  }_{t} \tag{28}
\]

where \( {\delta }_{p} \) is the position error, \( \left| {\delta }_{p}\right|  = \; \sqrt{{\left( {p}_{x} - {p}_{x}^{\text{ ref }}\right) }^{2} + {\left( {p}_{y} - {p}_{y}^{\text{ ref }}\right) }^{2}},\operatorname{sign}\left( {\delta }_{p}\right) \) is positive if the ego is on the left side of the reference path, or else is negative. \( {\delta }_{\phi } = \phi  - {\phi }^{\text{ ref }} \) is the error of heading angle, and \( {\delta }_{v} = {v}_{\text{ lon }} - {v}_{\text{ lon }}^{\text{ ref }} \) is the velocity error. The overall state design is illustrated in Fig. 3 The weighting matrices in the utility function are designed as \( Q = \operatorname{diag}\left( {{0.04},{0.04},{0.01},{0.01},{0.1},{0.02}}\right) \) , \( R = \operatorname{diag}\left( {{0.1},{0.005}}\right) \) . The predictive horizon \( T \) is set to be 25, which is 2.5s in practical.

其中\( {\delta }_{p} \)是位置误差，若自车在参考路径左侧，则\( \left| {\delta }_{p}\right|  = \; \sqrt{{\left( {p}_{x} - {p}_{x}^{\text{ ref }}\right) }^{2} + {\left( {p}_{y} - {p}_{y}^{\text{ ref }}\right) }^{2}},\operatorname{sign}\left( {\delta }_{p}\right) \)为正，否则为负。\( {\delta }_{\phi } = \phi  - {\phi }^{\text{ ref }} \)是航向角误差，\( {\delta }_{v} = {v}_{\text{ lon }} - {v}_{\text{ lon }}^{\text{ ref }} \)是速度误差。整体状态设计如图3所示。效用函数中的加权矩阵设计为\( Q = \operatorname{diag}\left( {{0.04},{0.04},{0.01},{0.01},{0.1},{0.02}}\right) \)，\( R = \operatorname{diag}\left( {{0.1},{0.005}}\right) \)。预测时域\( T \)设置为25，实际中为2.5秒。

![bo_d7lhveilb0pc73f2i630_7_243_721_534_536_0.jpg](images/bo_d7lhveilb0pc73f2i630_7_243_721_534_536_0.jpg)

Fig. 3: State design in the scenario.

图3:场景中的状态设计。

3) Constraint construction: Slightly different from the one in (1), we further refine the constraint in a way that represents the ego vehicle and each of the surrounding vehicles by two circles as illustrated by Fig. 4, where \( {r}_{\text{ veh }} \) and \( {r}_{\text{ ego }} \) are radii of circles of a vehicle and the ego. Then, in each time step, we impose four constraints for each vehicle between each center of the ego circle and that of the other vehicle rather than between only their CGs. Similar are the constraints between the ego and the road edge. Parameters for constraints: \( M = \operatorname{diag}\left( {1,1,0,0,0,0}\right) ,{D}_{\text{ veh }}^{\text{ safe }} = {r}_{\text{ veh }} + {r}_{\text{ ego }},{D}_{\text{ road }}^{\text{ safe }} = {r}_{\text{ ego }}, \) where \( {r}_{\text{ ego }} = {r}_{\text{ veh }} = {2.5}\mathrm{\;m} \) . For the traffic light constraint, we convert it to constraints between ego and vehicles by placing two virtual vehicles on the stop line, as shown in Fig. 4

3)约束构建:与(1)中的方法略有不同，我们进一步细化约束，用两个圆分别表示自车和每辆周围车辆，如图4所示，其中\( {r}_{\text{ veh }} \)和\( {r}_{\text{ ego }} \)分别是车辆和自车的圆半径。然后，在每个时间步，我们在自车圆中心和其他车辆圆中心之间为每辆车施加四个约束，而不是仅在它们的质心之间施加约束。自车与道路边缘之间的约束类似。约束参数:\( M = \operatorname{diag}\left( {1,1,0,0,0,0}\right) ,{D}_{\text{ veh }}^{\text{ safe }} = {r}_{\text{ veh }} + {r}_{\text{ ego }},{D}_{\text{ road }}^{\text{ safe }} = {r}_{\text{ ego }}, \)，其中\( {r}_{\text{ ego }} = {r}_{\text{ veh }} = {2.5}\mathrm{\;m} \)。对于交通灯约束，我们通过在停车线上放置两个虚拟车辆将其转换为自车与车辆之间的约束，如图4所示

4) Vehicle dynamics and prediction model: \( {F}_{\text{ ego }} \) has been shown in (3), where all the vehicles parameters are displayed in Table 1. Moreover, according to the type and position of the vehicle \( j, j \in  I \) , the turning rate \( {\omega }_{\text{ pred }}^{j} \) in the prediction model is determined, as shown in Table II

4)车辆动力学和预测模型:\( {F}_{\text{ ego }} \)已在(3)中展示，其中所有车辆参数列于表1。此外，根据车辆\( j, j \in  I \)的类型和位置，确定预测模型中的转弯速率\( {\omega }_{\text{ pred }}^{j} \)，如表二所示

5) Training settings: We implement the offline training Algorithm 2 in an asynchronous learning architecture proposed in [34]. For value function and policy, we use a multilayer perceptron (MLP) with 2 hidden layers, consisting of 256 units per layer, with Exponential Linear Units (ELU) each layer [35]. The Adam method [36] with a polynomial decay learning rate is used to update all the parameters. Specific hyperparameter settings are listed in Table III, We train 5 different runs with different random seeds on a single computer with a 2.4 GHz 50 core Inter Xeon CPU, with evaluations every 100 iterations.

5)训练设置:我们在[34]中提出的异步学习架构中实现离线训练算法2。对于价值函数和策略，我们使用具有2个隐藏层的多层感知器(MLP)，每层由256个单元组成，每层使用指数线性单元(ELU)[35]。使用具有多项式衰减学习率的Adam方法[36]更新所有参数。具体超参数设置列于表三，我们在一台配备2.4 GHz 50核英特尔至强CPU的计算机上使用不同随机种子进行5次不同的运行训练，每100次迭代进行评估。

![bo_d7lhveilb0pc73f2i630_7_1059_154_460_347_0.jpg](images/bo_d7lhveilb0pc73f2i630_7_1059_154_460_347_0.jpg)

Fig. 4: Design of the state constraints.

图4:状态约束设计。

TABLE I: Parameters for \( {F}_{\text{ ego }} \)

表一:\( {F}_{\text{ ego }} \)的参数

<table><tr><td>Parameter</td><td>Meaning</td><td>Value</td></tr><tr><td>\( {k}_{f} \)</td><td>Front wheel cornering stiffness</td><td>-155495 [N/rad]</td></tr><tr><td>\( {k}_{r} \)</td><td>Rear wheel cornering stiffness</td><td>-155495 [N/rad]</td></tr><tr><td>\( {L}_{f} \)</td><td>Distance from CG to front axle</td><td>1.19 [m]</td></tr><tr><td>\( {L}_{r} \)</td><td>Distance from CG to rear axle</td><td>1.46 [m]</td></tr><tr><td>\( m \)</td><td>Mass</td><td>1520 [kg]</td></tr><tr><td>\( {I}_{z} \)</td><td>Polar moment of inertia at CG</td><td>2640 [kg·m2]</td></tr><tr><td>\( {\Delta t} \)</td><td>System frequency</td><td>0.1 [s]</td></tr></table>

## C. Simulation results

## C. 仿真结果

Followed the Algorithm 1 the planned paths are shown in Fig. 5a. We also demonstrate the tracking and safety performances of different tasks during the training process in Fig. 5 indicated by \( {J}_{\text{ actor }} \) and \( {J}_{\text{ penalty }} \) respectively, and the value loss \( {J}_{\text{ critic }} \) to exhibit the performance of the value function. Along the training process, the policy loss, the penalty and the value loss decrease consistently for all the tasks, indicating an improving tracking and safety performance. Specially, the penalty and value loss decrease to zero approximately, proving the effectiveness of the proposed RL-based solver for constrained OCPs. In addition, the convergence speed, variance across different seeds, and the final performance vary with tasks. That is because the surrounding vehicles that have potential conflicts with the ego are different across tasks, leading to differences in task difficulty.

按照算法1，规划路径如图5a所示。我们还在图5中分别用\( {J}_{\text{ actor }} \)和\( {J}_{\text{ penalty }} \)展示了训练过程中不同任务的跟踪和安全性能，并用价值损失\( {J}_{\text{ critic }} \)展示价值函数的性能。在整个训练过程中，所有任务的策略损失、惩罚和价值损失都持续下降，表明跟踪和安全性能在提高。特别地，惩罚和价值损失近似降至零，证明了所提出的基于强化学习的约束最优控制问题求解器的有效性。此外，收敛速度、不同种子间的方差以及最终性能因任务而异。这是因为与自车有潜在冲突的周围车辆在不同任务中不同，导致任务难度不同。

In addition, we apply the trained policy of the left-turning task in the environment to visualize one typical case where a dense traffic and different phase of traffic light are included. Note that different from the training, we carry out all the simulation tests on a \( {2.90}\mathrm{{GHz}} \) Intel Core i9-8950HK CPU. As shown in Fig. 6, when the traffic light is red, the ego pulls up to avoid collision and to obey the rule. Then when the light turns green, the ego starts itself off to enter the junction, where it meets several straight-going vehicles from the opposite direction. Therefore, the ego chooses the upper path and slows down to try to bypass the first one. Notice that the safety shield works here to avoid potential collisions. After that, it speeds up to follow the middle path, the one with the lowest value in that case, with which it can go first and meanwhile avoid the right-turning vehicles so that the velocity tracking error can be largely reduced. The computing time in each step is under 10ms, making our method considerably fast to be applied in the real world.

此外，我们将左转任务训练好的策略应用于该环境，以可视化一个包含密集交通和不同交通灯相位的典型案例。注意，与训练不同，我们在一个\( {2.90}\mathrm{{GHz}} \)英特尔酷睿i9 - 8950HK CPU上进行所有仿真测试。如图6所示，当交通灯为红色时，自车停车以避免碰撞并遵守规则。然后当灯变绿时，自车启动进入路口，在那里它遇到来自相反方向的几辆直行车辆。因此，自车选择上方路径并减速试图绕过第一辆车。注意这里安全护盾起作用以避免潜在碰撞。之后，它加速跟随中间路径，在那种情况下中间路径的值最低，通过该路径它可以先行驶并同时避开右转车辆，从而可大幅降低速度跟踪误差。每一步的计算时间在10毫秒以内，这使得我们的方法在现实世界中应用相当快速。

TABLE II: Parameters for \( {F}_{\text{ pred }} \)

表二:\( {F}_{\text{ pred }} \)的参数

<table><tr><td colspan="2" rowspan="2">\( {\omega }_{\text{ pred }}^{j} \)</td><td colspan="2">Position reletive to the intersection</td></tr><tr><td>Out of</td><td>Within</td></tr><tr><td rowspan="3">Vehicle type</td><td>Left-turning</td><td>0</td><td>\( \frac{{v}_{\text{ lon }}^{j}/{26.875}}{0} \)</td></tr><tr><td>Straight-going</td><td>0</td><td></td></tr><tr><td>Right-turning</td><td>0</td><td>\( - {v}_{\text{ lon }}^{j}/{15.625} \)</td></tr></table>

TABLE III: Detailed hyperparameters.

表三:详细超参数

<table><tr><td>Hyperparameters</td><td>Value</td></tr><tr><td>Optimizer</td><td>Adam \( \left( {{\beta }_{1} = {0.9},{\beta }_{2} = {0.999}}\right) \)</td></tr><tr><td>Approximation function</td><td>MLP</td></tr><tr><td>Number of hidden layers</td><td>2</td></tr><tr><td>Number of hidden units</td><td>256</td></tr><tr><td>Nonlinearity of hidden layer</td><td>ELU</td></tr><tr><td>Replay buffer size</td><td>5e5</td></tr><tr><td>Batch size</td><td>1024</td></tr><tr><td>Policy learning rate</td><td>Linear decay 3e-4 \( \rightarrow \) 1e-5</td></tr><tr><td>Value learning rate</td><td>Linear decay 8e-4 \( \rightarrow \) 1e-5</td></tr><tr><td>Penalty amplifier \( c \)</td><td>1.1</td></tr><tr><td>Total iteration</td><td>200000</td></tr><tr><td>Update interval \( m \)</td><td>10000</td></tr><tr><td>Safety shield \( {n}_{ss} \)</td><td>5</td></tr><tr><td>Number of Actors</td><td>4</td></tr><tr><td>Number of Buffers</td><td>4</td></tr><tr><td>Number of Learners</td><td>30</td></tr></table>

![bo_d7lhveilb0pc73f2i630_8_145_725_732_760_0.jpg](images/bo_d7lhveilb0pc73f2i630_8_145_725_732_760_0.jpg)

Fig. 5: Results of static path planning and dynamic optimal tracking. (a) Planned paths for each task. (b) Tracking performance during training process. (c) Safety performance during training process. (d) Value loss during training process. For (b)-(d), The solid lines correspond to the mean and the shaded regions correspond to 95% confidence interval over 5 runs.

图5:静态路径规划和动态最优跟踪结果。(a) 每个任务的规划路径。(b) 训练过程中的跟踪性能。(c) 训练过程中的安全性能。(d) 训练过程中的值损失。对于(b)-(d)，实线对应均值，阴影区域对应5次运行的95%置信区间。

## D. Experiment 1: Performance of GEP

## D. 实验1:基因表达式编程(GEP)的性能

To verify the control precision and computing efficiency of our model-based solver on constrained optimal problem, we conduct comparison with the classic Model Predictive Control (MPC), which utilizes the receding horizon optimization online and can deal with constraints explicitly. Formally, the problem 1) is defined on one certain path, and thus MPC method solves the same number of problems as that of candidate paths and calculate cost function of each path respectively. Then optimal path is selected by the minimum cost function and its corresponding action is used as the input signal of ego vehicle. Here we adopt the Ipopt solver to obtain the exact solution of the constrained optimal control problem, which is an open and popular package to solve nonlinear optimization problem. Fig. 7 demonstrates the comparison of our algorithm on control effect and computation time. Results show that the optimal path of two methods are identical and the output actions, steer wheel and acceleration, have similar trends, which indicates our proposed algorithm can approximate the solution of MPC with a small error. However, there exists the obvious difference in computation time that our method can output the actions within \( {10}\mathrm{\;{ms}} \) while MPC will take 1000ms to perform that. Although MPC can find the optimal solution by its online optimization, its computation time also increases sharply with the number of constraints, probably violating the real-time requirements of autonomous driving.

为验证基于模型的求解器在约束最优问题上的控制精度和计算效率，我们与经典的模型预测控制(MPC)进行比较，MPC在线利用滚动时域优化，能明确处理约束。形式上，问题1)在某一条路径上定义，因此MPC方法求解的问题数量与候选路径数量相同，并分别计算每条路径的成本函数。然后通过最小成本函数选择最优路径，其相应的动作用作自车的输入信号。这里我们采用Ipopt求解器来获得约束最优控制问题的精确解，Ipopt是一个用于求解非线性优化问题的开源且流行的软件包。图7展示了我们的算法在控制效果和计算时间方面的比较。结果表明两种方法的最优路径相同，输出动作、方向盘和加速度具有相似趋势，这表明我们提出的算法能以较小误差逼近MPC的解。然而，在计算时间上存在明显差异，我们的方法能在\( {10}\mathrm{\;{ms}} \)内输出动作，而MPC执行该操作需要1000毫秒。虽然MPC能通过其在线优化找到最优解，但其计算时间也会随着约束数量的增加而急剧增加，可能违反自动驾驶的实时要求。

## E. Experiment 2: Comparison of driving performance

## E. 实验2:驾驶性能比较

We compare our method with a rule-based method which adopts A* algorithm to generate a feasible trajectory and a PID controller to track it [19], as well as a model-free RL method which uses a punish and reward system to learn a policy for maximizing the long-term reward (If the ego vehicle passes the intersection safely, a large positive reward 100 is given, otherwise -100 is given wherever a collision happens) [37]. We choose six indicators including computing time, comfort, travel efficiency, collisions, failure rate and driving compliance to evaluate the three algorithms. Comfort is reflected by the mean root square of lateral and longitudinal acceleration, i.e., \( {I}_{\text{ comfort }} = {1.4}\sqrt{\left( {a}_{x}^{2}\right)  + \left( {a}_{y}^{2}\right) } \) . Travel efficiency is evaluated by the average time used to pass the intersection. Failure rate means the accumulated times of that decision signal is generated for more than 1 seconds and driving compliance shows times of breaking red light. The results of 100 times simulation are shown in Table IV Rule-based method is more likely to stop and wait for other vehicles, leading to higher passing time but better safety performance. However, it tends to take much more time to give a control signal in the dense traffic flow and suffers the highest failure rate. Model-free RL is eager to achieve the goal, but incurs the most collisions and decision incompliance due to its lack of safety guarantee. Benefiting from the framework design, the computational efficiency of our method remains as fast as the model-free approach, and the driving performance such as safety, compliance and failure rate, is better than the other two approaches.

我们将我们的方法与一种基于规则的方法进行比较，该方法采用A*算法生成可行轨迹并使用PID控制器进行跟踪[19]，以及一种无模型强化学习方法，该方法使用惩罚和奖励系统来学习策略以最大化长期奖励(如果自车安全通过路口，给予大的正奖励100，否则无论何处发生碰撞都给予 -100)[37]。我们选择计算时间、舒适性、行驶效率、碰撞次数、故障率和驾驶合规性六个指标来评估这三种算法。舒适性由横向和纵向加速度的均方根反映，即\( {I}_{\text{ comfort }} = {1.4}\sqrt{\left( {a}_{x}^{2}\right)  + \left( {a}_{y}^{2}\right) } \)。行驶效率通过通过路口所用的平均时间来评估。故障率指决策信号生成超过1秒的累计次数，驾驶合规性表示闯红灯次数。100次仿真结果如表IV所示。基于规则的方法更倾向于停车等待其他车辆，导致通过时间更长但安全性能更好。然而，在密集交通流中给出控制信号往往需要更多时间，且故障率最高。无模型强化学习急于实现目标，但由于缺乏安全保障，导致碰撞次数最多且决策不符合要求。受益于框架设计，我们方法的计算效率与无模型方法一样快，并且在安全性、合规性和故障率等驾驶性能方面优于其他两种方法。

![bo_d7lhveilb0pc73f2i630_9_133_176_1538_684_0.jpg](images/bo_d7lhveilb0pc73f2i630_9_133_176_1538_684_0.jpg)

Fig. 6: Visualization of one typical episode driven by the trained policy.

图6:由训练好的策略驱动的一个典型情节的可视化。

![bo_d7lhveilb0pc73f2i630_9_163_992_696_675_0.jpg](images/bo_d7lhveilb0pc73f2i630_9_163_992_696_675_0.jpg)

Fig. 7: Comparison with MPC. (a) Front wheel angle. (b) Acceleration. (c) Optimal path. (d) Computing time

图7:与MPC的比较。(a) 前轮角度。(b) 加速度。(c) 最优路径。(d) 计算时间

## F. Experiment 3: Application of distributed control

## F. 实验3:分布式控制的应用

We also apply our trained policies in multiple vehicles for distributed control. Our method yields a surprisingly good performance by showing a group of complex and intelligent driving trajectories (Fig. 8), which demonstrates the potential of the proposed method to be extended to the distributed control of large-scale connected vehicles. More videos are available online (https://youtu.be/J8aRgcjJukQ).

我们还将训练好的策略应用于多辆车进行分布式控制。我们的方法通过展示一组复杂而智能的驾驶轨迹产生了惊人的良好性能(图8)，这证明了所提出方法扩展到大规模联网车辆分布式控制的潜力。更多视频可在网上获取(https://youtu.be/J8aRgcjJukQ)。

TABLE IV: Comparison of driving performance

表四:驾驶性能比较

<table><tr><td></td><td>IDC (Ours)</td><td>Rule-based</td><td>Model-free RL</td></tr><tr><td colspan="4">Computing time [ms]</td></tr><tr><td>Upper-quantile</td><td>5.81</td><td>73.99</td><td>4.91</td></tr><tr><td>Standard deviation</td><td>0.60</td><td>36.59</td><td>0.65</td></tr><tr><td>Comfort index</td><td>1.84</td><td>1.41</td><td>3.21</td></tr><tr><td>Time to pass [s]</td><td>\( {7.86}\left( {\pm {3.52}}\right) \)</td><td>24.4(±16.48)</td><td>6.73(±3.32)</td></tr><tr><td>Collisions</td><td>0</td><td>0</td><td>31</td></tr><tr><td>Failure Rate</td><td>0</td><td>13</td><td>0</td></tr><tr><td>Decision Compliance</td><td>0</td><td>0</td><td>17</td></tr></table>

## VI. TEST ON REAL-WORLD ROADS

## 六、实际道路测试

## A. Scenario and equipment

## A. 场景与设备

In the real-world test, we choose an intersection of two-way streets located at \( \left( {{31}^{ \circ  }{08}^{\prime }{13}^{\prime \prime }\mathrm{N},{120}^{ \circ  }{35}^{\prime }{06}^{\prime \prime }\mathrm{E}}\right) \) , shown in Fig. 9a. The east-west street has a eight-lane dual carriageway from both directions, while the north-south street has a only four-lane dual carriageway. The detailed size and functionality of each lane are illustrated in Fig. 9b. To comply with legal requirements, we did not utilize the traffic flow and traffic signals of the real intersection. Instead, these traffic elements are designed and provided by SUMO. The experiment vehicle is CHANA CS55 equipped with RTK GPS, which realizes precise localization of the ego vehicle. In each time step, the ego states gathered from the CAN bus and the RTK are mapped into SUMO traffic to obtain the current traffic states including the states of surrounding vehicles and traffic signals. Then they both are sent to the industrial computer, where the trained policy and value functions are embedded. The computer is KMDA-3211 with a 2.6 GHz Intel Core I5-6200U CPU. Processed by our online algorithm, the safe actions including the steering angle and the expected acceleration are then delivered from the CAN bus to the real vehicle for its real time control. The experiment settings are illustrated in Fig. 10 Similar to section V, the ego vehicle enters the intersection from south, and is required to complete the same tasks (turn left, go straight, and turn right) under the signal control and a dense traffic flow.

在实际测试中，我们选择了位于\( \left( {{31}^{ \circ  }{08}^{\prime }{13}^{\prime \prime }\mathrm{N},{120}^{ \circ  }{35}^{\prime }{06}^{\prime \prime }\mathrm{E}}\right) \)的双向街道的一个十字路口，如图9a所示。东西向街道双向各有一条八车道的双车道公路，而南北向街道只有一条四车道的双车道公路。每条车道的详细尺寸和功能如图9b所示。为了符合法律要求，我们没有使用实际十字路口的交通流和交通信号。相反，这些交通元素由SUMO设计和提供。实验车辆是配备了RTK GPS的长安CS55，它实现了自车的精确定位。在每个时间步，从CAN总线和RTK收集的自车状态被映射到SUMO交通中，以获得当前的交通状态，包括周围车辆和交通信号的状态。然后它们都被发送到工业计算机，在那里嵌入了经过训练的策略和价值函数。该计算机是KMDA - 3211，配备2.6 GHz英特尔酷睿I5 - 6200U CPU。通过我们的在线算法处理后，包括转向角和预期加速度在内的安全动作然后从CAN总线传递到实际车辆进行实时控制。实验设置如图10所示。与第五节类似，自车从南进入十字路口，并被要求在信号控制和密集交通流下完成相同的任务(左转、直行和右转)。

![bo_d7lhveilb0pc73f2i630_10_127_176_1549_1104_0.jpg](images/bo_d7lhveilb0pc73f2i630_10_127_176_1549_1104_0.jpg)

Fig. 8: Demonstration of the distributed control carried out by the trained policies.

图8:训练策略进行的分布式控制演示。

![bo_d7lhveilb0pc73f2i630_10_142_1678_740_443_0.jpg](images/bo_d7lhveilb0pc73f2i630_10_142_1678_740_443_0.jpg)

Fig. 9: The intersection for the real test.

图9:实际测试的十字路口。

![bo_d7lhveilb0pc73f2i630_10_945_1381_683_569_0.jpg](images/bo_d7lhveilb0pc73f2i630_10_945_1381_683_569_0.jpg)

Fig. 10: Diagram of the real-world road test.

图10:实际道路测试示意图。

## B. Experiment 1: Functionality verification

## B. 实验1:功能验证

This experiment aims to verify the functionality of the IDC framework under different tasks and scenarios. In total, nine runs were carried out, three for each task. In each run, the ego vehicle is initialized before the south entrance with random states, meanwhile, the surrounding vehicles and signals are also initialized randomly. Following the diagram in Fig. 10, the run keeps going on until the ego passes the intersection successfully, i.e., without colliding with obstacles or breaking traffic rules. The diversity among different runs is guaranteed by using different random seeds. All the videos are available online (https://youtu.be/adqjor5KXxQ).

本实验旨在验证IDC框架在不同任务和场景下的功能。总共进行了九次运行，每个任务三次。在每次运行中，自车在南入口前用随机状态初始化，同时，周围车辆和信号也随机初始化。按照图10中的示意图，运行一直持续到自车成功通过十字路口，即不与障碍物碰撞或违反交通规则。通过使用不同的随机种子保证了不同运行之间的多样性。所有视频可在网上获取(https://youtu.be/adqjor5KXxQ)。

![bo_d7lhveilb0pc73f2i630_11_197_182_1407_636_0.jpg](images/bo_d7lhveilb0pc73f2i630_11_197_182_1407_636_0.jpg)

Fig. 11: Featured time steps of the left run.

图11:左转运行的特色时间步。

![bo_d7lhveilb0pc73f2i630_11_160_945_701_707_0.jpg](images/bo_d7lhveilb0pc73f2i630_11_160_945_701_707_0.jpg)

Fig. 12: Key parameters in the left run.

图12:左转运行中的关键参数。

Since the left-turning is the most complex task with the most potential interactions with surrounding vehicles, we visualize one of the left runs by snapshotting its featured time steps shown in Fig. 11 and drawing the key parameters in Fig. 12 At the beginning, the ego pulls up before the stop line, waiting for the green light (Fig. 11a). When that comes, the ego accelerates into the intersection to reduce the velocity tracking error (Fig. 11b). In the center of the intersection, it encounters a straight-going vehicle with high speed from the opposite direction. In order to avoid collision, the ego slows itself down and switches to the path 4 , with which it is able to bypass the vehicle from back (Fig. 11c). However, another straight-going vehicle comes over after the previous one passes through, but with a relative low speed. This time, the ego chooses to accelerate to pass first. Interestingly, as the vehicle approaches, the optimal path is automatically selected away from it, i.e., changing from the path 4 to the path 3 and finally the path 2, to minimize the tracking errors (Fig. 11c and Fig. 11d). Following the path 2, the ego finally passes the intersection successfully. The computing time of all step is within \( {15}\mathrm{\;{ms}} \) , showing the superior of our method in terms of the online computing efficiency.

由于左转是最复杂的任务，与周围车辆的潜在交互最多，我们通过拍摄图11所示的其特色时间步并绘制图12中的关键参数来可视化其中一次左转运行。开始时，自车在停止线前停车，等待绿灯(图11a)。绿灯亮起时，自车加速进入十字路口以减小速度跟踪误差(图11b)。在十字路口中心，它遇到一辆从相反方向高速驶来的直行车辆。为了避免碰撞，自车减速并切换到路径4，通过该路径它能够从后方绕过该车辆(图11c)。然而，在前一辆车通过后，另一辆直行车辆驶来，但速度相对较低。这次，自车选择加速先通过。有趣的是，当车辆靠近时，最优路径会自动选择远离它，即从路径4变为路径3，最后变为路径2，以最小化跟踪误差(图11c和图11d)。沿着路径2，自车最终成功通过十字路口。所有步骤的计算时间在\( {15}\mathrm{\;{ms}} \)内，显示了我们的方法在在线计算效率方面的优越性。

## C. Experiment 2: Robustness to noise

## C. 实验2:对噪声的鲁棒性

This experiment aims to compare the driving performance under different levels of noises added manually to verify the robustness of the trained policies. Referring to [38], we take similar measure to divide the noises into 7 levels, i.e. 0-6, where all the noises are in form of Gaussian white noise with different variances varying with the level and are applied in several dimensions of RL states, as shown in Table V.

本实验旨在比较在手动添加不同水平噪声的情况下的驾驶性能，以验证训练策略的鲁棒性。参考[38]，我们采取类似的措施将噪声分为7个级别，即0 - 6，其中所有噪声均为高斯白噪声形式，其方差随级别不同而变化，并应用于强化学习状态的几个维度，如表五所示。

We choose the left-turning task to perform seven experiments, one for each noise level in the Table V, to show its influence on the effect of the proposed framework. For each noise level, i.e., each experiment, we make statistical analysis on the parameters related to vehicle stability, namely the yaw rate and lateral speed, and the control quantities, i.e., steering angle and acceleration, as shown in Fig. 13 Our method is rather robust to low level noises (0-3) in which the distribution parameters including the median value, the standard variance, the quantile values and the bounds have no significant change. However, these parameters, especially the variance and the bounds, are inevitably enlarged if we add stronger noise. The fluctuations of the lateral velocity and the yaw rate are mainly caused by the sensitivity of the steering wheel, because large noises tend to yield large variance of the steering angle, which further leads to the swing of the vehicle body. But nevertheless the stability bounds always remain in a reasonable range, proving the robustness of the proposed method.

我们选择左转任务进行七次实验，对应表五中的每个噪声水平进行一次实验，以展示其对所提框架效果的影响。对于每个噪声水平，即每次实验，我们对与车辆稳定性相关的参数(即横摆率和横向速度)以及控制量(即转向角和加速度)进行统计分析，如图13所示。我们的方法对低水平噪声(0 - 3)具有较强的鲁棒性，其中包括中位数、标准差、分位数和边界在内的分布参数没有显著变化。然而，如果添加更强的噪声，这些参数，尤其是方差和边界，将不可避免地增大。横向速度和横摆率的波动主要由方向盘的灵敏度引起，因为大噪声往往会导致转向角的方差增大，进而导致车身摆动。但稳定性边界始终保持在合理范围内，证明了所提方法的鲁棒性。

TABLE V: Noise level and the corresponding standard deviation.

表五:噪声水平及相应标准差。

<table><tr><td>Noise level</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>\( {\delta }_{p}\left\lbrack  \mathrm{\;m}\right\rbrack \)</td><td>0</td><td>0.017</td><td>0.033</td><td>0.051</td><td>0.068</td><td>0.085</td><td>0.102</td></tr><tr><td>\( {\delta }_{\phi }\left\lbrack  {}^{ \circ  }\right\rbrack \)</td><td>0</td><td>0.017</td><td>0.033</td><td>0.051</td><td>0.068</td><td>0.085</td><td>0.102</td></tr><tr><td>\( {p}_{x}^{j},{p}_{y}^{j}\left\lbrack  \mathrm{\;m}\right\rbrack \)</td><td>0</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td></tr><tr><td>\( {v}_{\text{ lon }}^{j}\left\lbrack  {\mathrm{\;m}/\mathrm{s}}\right\rbrack \)</td><td>0</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td></tr><tr><td>\( {\phi }^{j}\left\lbrack  {}^{ \circ  }\right\rbrack \)</td><td>0</td><td>1.4</td><td>2.8</td><td>4.2</td><td>5.6</td><td>7.0</td><td>8.4</td></tr></table>

![bo_d7lhveilb0pc73f2i630_12_138_916_743_664_0.jpg](images/bo_d7lhveilb0pc73f2i630_12_138_916_743_664_0.jpg)

Fig. 13: States and actions in different noise level.

图13:不同噪声水平下的状态和动作。

## D. Experiment 3: Robustness to human disturbance

## D. 实验3:对人为干扰的鲁棒性

The experiment is to verify the ability of our method to cope with human disturbance. We also use a left-turning case, in which we perform two times of human intervention on the steer wheel, and then switch to the autonomous driving mode. We draw the states of the ego vehicle in Fig. 14 where the colored region is when the human disturbance is acted on. The first one is acted on 10s, when the ego just enters the crossroad and we turn the steering wheel left to \( {100}^{ \circ  } \) from \( {0}^{ \circ  } \) to make the ego head to the left. After that the driving system turn the steering wheel right immediately to correct the excessive ego heading. The second one happens at 16s, when the ego is turning to the left to pass the crossroad. We turn the steering wheel right from \( {90}^{ \circ  } \) to \( {0}^{ \circ  } \) to interrupt the process. After the take-over, the driving system is able to turn the steering wheel left to \( {240}^{ \circ  } \) right away to continue to complete the turn left operation. Results show that the proposed method is capable of dealing with the abrupt human disturbance on the steering wheel by quick responses to the interrupted state after taking over.

该实验旨在验证我们的方法应对人为干扰的能力。我们同样使用左转场景，在该场景中对方向盘进行两次人为干预，然后切换到自动驾驶模式。我们在图14中绘制了自车的状态，其中彩色区域表示施加人为干扰的时刻。第一次干预发生在10秒时，此时自车刚进入十字路口，我们将方向盘从\( {0}^{ \circ  } \)向左转动到\( {100}^{ \circ  } \)，使自车向左行驶。之后驱动系统立即将方向盘向右转以纠正自车过度左转的方向。第二次干预发生在16秒时，此时自车正在左转通过十字路口。我们将方向盘从\( {90}^{ \circ  } \)向右转到\( {0}^{ \circ  } \)以中断该过程。接管后，驱动系统能够立即将方向盘向左转动到\( {240}^{ \circ  } \)以继续完成左转操作。结果表明，所提方法能够通过对接管后的中断状态做出快速响应，应对方向盘上的突发人为干扰。

![bo_d7lhveilb0pc73f2i630_12_913_426_746_669_0.jpg](images/bo_d7lhveilb0pc73f2i630_12_913_426_746_669_0.jpg)

Fig. 14: States and actions under human disturbance.

图14:人为干扰下的状态和动作。

## VII. CONCLUSION

## VII. 结论

In this paper, we propose integrated decision and control (IDC) framework for automated vehicles, for the purpose of building an interpretable learning system with high online computing efficiency and applicability among different driving tasks and scenarios. The framework decomposes the driving task into the static path planning and the dynamic optimal tracking hierarchically. The former is in charge of generating multiple paths only considering static constraints, which are then sent to the latter to be selected and tracked. The latter first formulates the selecting and tracking problem as constrained OCPs mathematically to take dynamic obstacles into consideration, and then solves it offline by a model-based RL algorithm we propose to seek an approximate solution of an OCP in form of neural networks. Notably, these solved approximation functions, namely value and policy, have a natural correspondence to the selecting and tracking problems, which originates the interpretability. Finally, the value and policy functions are used online instead, releasing the heavy computation due to online optimizations. We verify our framework in both simulation and in a real world intersection. Results show that our method has an order of magnitude higher online computing efficiency compared with the traditional rule-based method. In addition, it yields better driving performance in terms of traffic efficiency and safety, and shows great interpretability and adaptability among different driving tasks. In the future, we will place efforts on designing more general state represen-

在本文中，我们为自动驾驶车辆提出了集成决策与控制(IDC)框架，旨在构建一个具有高在线计算效率且适用于不同驾驶任务和场景的可解释学习系统。该框架将驾驶任务分层分解为静态路径规划和动态最优跟踪。前者负责仅考虑静态约束生成多条路径，然后将这些路径发送给后者进行选择和跟踪。后者首先将选择和跟踪问题数学公式化为受约束的最优控制问题(OCP)以考虑动态障碍物，然后通过我们提出的基于模型的强化学习算法离线求解，以寻求神经网络形式的OCP近似解。值得注意的是，这些求解得到的近似函数，即价值函数和策略函数，与选择和跟踪问题具有自然的对应关系，这产生了可解释性。最后，在线使用价值函数和策略函数，避免了由于在线优化带来的繁重计算。我们在模拟和真实世界的十字路口对我们的框架进行了验证。结果表明，与传统的基于规则的方法相比，我们的方法在线计算效率提高了一个数量级。此外，在交通效率和安全性方面，它具有更好的驾驶性能，并且在不同驾驶任务中表现出很强的可解释性和适应性。未来，我们将致力于设计更通用的状态表示。