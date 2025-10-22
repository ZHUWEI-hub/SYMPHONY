# from collections import defaultdict
#
# from networkx.classes import non_edges
#
#
# class Node:
#     def __init__(self, state, question, parent=None):
#         self.state = {'thought': '', 'action': '', 'observation': ''} if state is None else state
#         self.parent = parent
#         self.question = question
#         self.children = []
#         self.visits = 0
#         self.value = 0
#         self.depth = 0 if parent is None else parent.depth + 1
#         self.is_terminal = False
#         self.reward = 0
#         self.exhausted = False  # If all children are terminal
#         self.em = 0  # Exact match, evaluation metric  那你他吗环境给的是True和False
#
#
#     # def uct(self):
#     #     if self.visits == 0:
#     #         return self.value
#     #     return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
#
#     def __str__(self):
#         return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, thought={self.state['thought']}, action={self.state['action']}, observation={self.state['observation']}, observation={self.question})"
#
#     def to_dict(self):
#         return {
#             'state': self.state,
#             'question': self.question,
#             'parent': self.parent.to_dict() if self.parent else None,
#             'children': [child.to_dict() for child in self.children],
#             'visits': self.visits,
#             'value': self.value,
#             'depth': self.depth,
#             'is_terminal': self.is_terminal,
#             'reward': self.reward,
#             'em': self.em,
#         }
# n = Node(state=None,question='1')
# n2 = Node(state=None,question='2')
# n3 = Node(state=None,question='1')
# new_states = [n,n2,n3]
# index_node = {}
#
# # same_count = 0
# # for child in new_states:
# # 创建分组字典：key=标准action，value=节点列表
# action_groups = defaultdict(list)
# for i in range(0, len(new_states)):
#     index_node[new_states[i]] = i
#     action_groups[new_states[i].question].append(new_states[i])
# print(index_node)
# print(action_groups)
# print(index_node[action_groups['1'][0]])
# print(index_node[action_groups['2'][0]])
# print(index_node[action_groups['1'][1]])