import json

data_dir = 'input.json'

def dfs(graph, i, j):
    if i < 0 or i >= len(graph) or j < 0 or j >= len(graph[0]):
        return 0
    if graph[i][j]==0:
        return 0
    graph[i][j]=0
    dfs(graph, i, j-1)
    dfs(graph, i, j+1)
    dfs(graph, i-1, j)
    dfs(graph, i+1, j)

    return 1

def count_island(graph):
    cnt = 0
    # print('i', len(graph), len(graph[0]))
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            cnt += dfs(graph, i, j)
    return cnt


if __name__ == '__main__':
    json_data = json.load(open(data_dir))
    for test_case in json_data:
        # print(json_data[test_case])
        count = count_island(json_data[test_case])
        print(f'number of island found {count}')
        # break
