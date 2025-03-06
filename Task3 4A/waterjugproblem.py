from collections import deque

def water_jug_dfs(c1,c2,goal):
    stack =[(0,0)] 
    visited =set()
    path = []
    
    while stack:
        jug1,jug2 = stack.pop()
        if (jug1,jug2) in visited:
            continue
        visited.add((jug1,jug2))
        path.append((jug1,jug2))
        
        if jug1 == goal or jug2 == goal:
            print("solution found!:")
            for step in path:
                print(step)
            return True
        rules = [
            (c1, jug2),                                        #Fill jug1
            (jug1, c2),                                        #Fill jug2
            (0, jug2),                                         #Empty jug1
            (jug1, 0),                                         #Empty jug2
            (max(0,jug1 - (c2 - jug2)),min(c2, jug1 + jug2)),  #Pour jug1 →jug2(until jug2 is full or jug1 is empty)
            (min(c1,jug1 + jug2),max(0,jug2 - (c1 - jug1)))    #Pour jug2 →jug1(until jug1 is full or jug2 is empty)
        ]
        for rule in rules:
            if rule not in visited:
                stack.append(rule)
    print("No solution found.")
    return False
jug1Capacity = 4
jug2Capacity = 3
goal = 2
water_jug_dfs(jug1Capacity,jug2Capacity,goal)
