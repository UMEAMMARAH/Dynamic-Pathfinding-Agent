import pygame
import sys
import math
import heapq
import random
import time

pygame.init()

# ==============================
# CONFIG
# ==============================

WIDTH = 900
HEIGHT = 700
PANEL_WIDTH = 250
GRID_WIDTH = WIDTH - PANEL_WIDTH
ROWS = 20
COLS = 20
CELL_SIZE = GRID_WIDTH // COLS

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,255,0)
CYAN = (0,255,255)
PURPLE = (150,0,150)
GRAY = (200,200,200)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dynamic Pathfinding Agent")

font = pygame.font.SysFont("Arial", 18)

# ==============================
# GRID
# ==============================

def create_grid():
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]

grid = create_grid()

start = (0, 0)
goal = (ROWS-1, COLS-1)

grid[start[0]][start[1]] = "S"
grid[goal[0]][goal[1]] = "G"

# ==============================
# HEURISTICS
# ==============================

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

current_heuristic = "manhattan"
current_algorithm = "astar"

# ==============================
# SEARCH ALGORITHMS
# ==============================

def get_neighbors(node):
    r, c = node
    neighbors = []
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            if grid[nr][nc] != 1:
                neighbors.append((nr,nc))
    return neighbors


def search(start_node):

    open_list = []
    heapq.heapify(open_list)

    g = {start_node: 0}
    parent = {}
    visited = set()
    frontier = set()

    if current_heuristic == "manhattan":
        h_func = manhattan
    else:
        h_func = euclidean

    if current_algorithm == "astar":
        f = g[start_node] + h_func(start_node, goal)
    else:
        f = h_func(start_node, goal)

    heapq.heappush(open_list, (f, start_node))
    frontier.add(start_node)

    nodes_expanded = 0
    start_time = time.time()

    while open_list:
        _, current = heapq.heappop(open_list)
        frontier.discard(current)

        if current == goal:
            end_time = time.time()
            return reconstruct_path(parent, current), visited, frontier, nodes_expanded, (end_time-start_time)*1000

        visited.add(current)
        nodes_expanded += 1

        for neighbor in get_neighbors(current):

            tentative_g = g[current] + 1

            if neighbor not in g or tentative_g < g[neighbor]:

                parent[neighbor] = current
                g[neighbor] = tentative_g

                if current_algorithm == "astar":
                    f_val = tentative_g + h_func(neighbor, goal)
                else:
                    f_val = h_func(neighbor, goal)

                heapq.heappush(open_list, (f_val, neighbor))
                frontier.add(neighbor)

    return None, visited, frontier, nodes_expanded, 0


def reconstruct_path(parent, node):
    path = []
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

# ==============================
# RANDOM OBSTACLES
# ==============================

def generate_random_obstacles(density=0.3):
    global grid
    grid = create_grid()

    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) != start and (r,c) != goal:
                if random.random() < density:
                    grid[r][c] = 1

    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"

# ==============================
# DRAWING
# ==============================

def draw_grid(path=None, visited=None, frontier=None):

    for r in range(ROWS):
        for c in range(COLS):

            x = c * CELL_SIZE
            y = r * CELL_SIZE

            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            color = WHITE

            if grid[r][c] == 1:
                color = BLACK

            if visited and (r,c) in visited:
                color = BLUE

            if frontier and (r,c) in frontier:
                color = YELLOW

            if path and (r,c) in path:
                color = GREEN

            if (r,c) == start:
                color = CYAN

            if (r,c) == goal:
                color = PURPLE

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)


def draw_panel(nodes, cost, exec_time):

    pygame.draw.rect(screen, (230,230,230), (GRID_WIDTH, 0, PANEL_WIDTH, HEIGHT))

    texts = [
        f"Algorithm: {current_algorithm}",
        f"Heuristic: {current_heuristic}",
        f"Nodes Expanded: {nodes}",
        f"Path Cost: {cost}",
        f"Exec Time (ms): {round(exec_time,2)}",
        "",
        "Controls:",
        "SPACE: Start Search",
        "R: Random Map",
        "D: Toggle Dynamic",
        "H: Switch Heuristic",
        "A: Switch Algorithm"
    ]

    y = 20
    for t in texts:
        img = font.render(t, True, (0,0,0))
        screen.blit(img, (GRID_WIDTH + 10, y))
        y += 30

# ==============================
# MAIN LOOP
# ==============================

dynamic_mode = False
current_path = []
agent_position = start

nodes_expanded = 0
path_cost = 0
execution_time = 0

clock = pygame.time.Clock()

running = True

while running:

    clock.tick(60)
    screen.fill(WHITE)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if x < GRID_WIDTH:
                c = x // CELL_SIZE
                r = y // CELL_SIZE
                if (r,c) != start and (r,c) != goal:
                    grid[r][c] = 1 if grid[r][c] == 0 else 0

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                generate_random_obstacles(0.3)

            if event.key == pygame.K_h:
                current_heuristic = "euclidean" if current_heuristic=="manhattan" else "manhattan"

            if event.key == pygame.K_a:
                current_algorithm = "gbfs" if current_algorithm=="astar" else "astar"

            if event.key == pygame.K_d:
                dynamic_mode = not dynamic_mode

            if event.key == pygame.K_SPACE:
                current_path, visited, frontier, nodes_expanded, execution_time = search(start)
                if current_path:
                    path_cost = len(current_path)-1

    # Dynamic obstacle spawning
    if dynamic_mode and current_path:
        if random.random() < 0.02:
            r = random.randint(0,ROWS-1)
            c = random.randint(0,COLS-1)
            if (r,c) not in current_path and (r,c) != start and (r,c) != goal:
                grid[r][c] = 1

    draw_grid(current_path)
    draw_panel(nodes_expanded, path_cost, execution_time)

    pygame.display.flip()

pygame.quit()
sys.exit()