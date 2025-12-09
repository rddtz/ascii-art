import cv2
import numpy as np
import math
import random
import copy
from tqdm import tqdm
import asciiVectorize as asciiV
import asciiAISS as asciiA
from skimage.draw import line

def VectorsAngle(v1, v2):

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)
    # angulo entre dois vetores

def IntersecLinha(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y):

    # Calculates the intersection point of two polylines (p0-p1) and (p2-p3).

    s1x = p0x - p1x
    s1y = p0y - p1y
    s2x = p2x - p3x
    s2y = p2y - p3y

    dx = p0x * p1y - p1x * p0y
    dy = p2x * p3y - p2y * p3x

    # Check for parallel lines to avoid div by zero
    div = s1x * s2y - s1y * s2x
    if div == 0: # denominador e zero
        return -1, -1

    x = (dx * s2x - dy * s1x) / div
    y = (dx * s2y - dy * s1y) / div

    tx1 = min(p0x, p1x); tx2 = max(p0x, p1x)
    ty1 = min(p0y, p1y); ty2 = max(p0y, p1y)
    tx3 = min(p2x, p3x); tx4 = max(p2x, p3x)
    ty3 = min(p2y, p3y); ty4 = max(p2y, p3y)

    if tx1 <= x <= tx2 and ty1 <= y <= ty2 and tx3 <= x <= tx4 and ty3 <= y <= ty4:
        return x, y
    else:
        return -1, -1 # não houve intersecção


def PontoMaisProximo(line, x_out, y_out):
    # Finds the closest point on a line to the point

    if not len(line):
        return 0, 0

    best_x, best_y = line[0]
    min_dist_sq = float('inf')

    # itera sobre toda a linha
    for i in range(len(line) - 1):

        x1, y1 = line[i]
        x2, y2 = line[i+1]

        px = x2 - x1
        py = y2 - y1
        norm_sq = px * px + py * py

        if norm_sq == 0:
            u = 0
        else:
            u = ((x_out - x1) * px + (y_out - y1) * py) / float(norm_sq) # projecao

        if u > 1: u = 1
        elif u < 0: u = 0

        x = x1 + u * px
        y = y1 + u * py

        # best result found so far
        dist_sq = (x - x_out)**2 + (y - y_out)**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_x, best_y = x, y

    return math.floor(best_x), int(best_y)


def PontosMaisProximos(polylines, point, degree):

    # Ray casting
    # Iterates through all polylines and all points (line segments) in those polylines.

    points = []
    res = set()

    # Cast rays in circle
    for i in range(0, 360, degree):

        # Convert degrees to radians for Python math functions
        rad = (i / 180.0) * math.pi
        x = int(math.sin(rad) * 150) + point[0]
        y = int(math.cos(rad) * 150) + point[1]

        min_x = 9999
        min_s = None # Will hold the specific link [p1, p2] hit

        # iterate over all polylines
        for j in range(len(polylines)):
            poly = polylines[j]

            # iterate over all edges in the current polyline
            for k in range(len(poly) - 1):
                p_start = poly[k]
                p_end = poly[k+1]

                # check intersection
                x_int, y_int = IntersecLinha(point[0], point[1], x , y,
                                                 p_start[0], p_start[1],
                                                 p_end[0], p_end[1])

                if x_int < 0 and y_int < 0:
                    continue

                dist = abs(x_int - point[0]) # or hypotenuse distance
                if dist < min_x:
                    min_x = dist
                    # Store just the segment that was hit
                    min_s = [p_start, p_end]

        if min_s:
            flat = np.array(min_s, dtype=str).flatten() # 1D array of strings
            res.add(','.join(flat))

    # Reconstruct the lines (made of two points) found
    found_links = [np.array(r.split(','), dtype=int).reshape(2, 2) for r in res]

    for link in found_links:
        n_point = PontoMaisProximo(link, point[0], point[1])
        # Avoid adding the point itself if the ray started exactly on a line
        if point[0] != n_point[0] or point[1] != n_point[1]:
            points.append(n_point)

    return points


def LocalDeform(A, B, C, args):

    # Calculates cost of deforming segment AB to AC.
    # Penalizes changing angle (Theta) and changing length (r).

    # empirically chosen = não mudarei :)
    l1 = 8.0 / math.pi
    l2 = 2.0 / min(args.tw, args.th)
    l3 = 0.5

    # Angular deformation cost
    V_theta = math.exp(math.fabs(l1 * VectorsAngle((B[0] - A[0], B[1] - A[1]), (C[0] - A[0], C[1] - A[1]))))

    # Length deformation cost
    r1 = DistEuclidiana(A, B)
    r2 = DistEuclidiana(A, C)
    diff = math.exp(l2 * math.fabs(r1 - r2))

    if min(r1, r2) == 0: prop = 0
    else: prop = math.exp(l3 * max(r1, r2) / min(r1, r2))

    V_r = max(diff, prop)
    return max(V_theta, V_r)


def Deform(polylines, chosed, old_pos, new_pos, args):

    if chosed[0] == new_pos[0] and chosed[1] == new_pos[1]:
        return -1 # Invalid move

    d_l = LocalDeform(chosed, old_pos, new_pos, args)

    # Midpoints
    Pa = (int(old_pos[0] + chosed[0]) // 2, int(old_pos[1] + chosed[1]) // 2)
    Pa_ = (int(new_pos[0] + chosed[0]) // 2, int(new_pos[1] + chosed[1]) // 2)

    l = []
    d_a = 0
    total_length = 0

    # Check deformation relative to points next to it
    points = PontosMaisProximos(polylines, Pa, 10)
    for i, p in enumerate(points):
        if (Pa[0] == p[0] and Pa[1] == p[1]) or (Pa_[0] == p[0] and Pa_[1] == p[1]):
            return -1 # Collision detected
        l.append(DistEuclidiana(Pa, p))
        total_length += l[-1]

    if total_length == 0:
        return d_l

    for i, p in enumerate(points):
        d_a += LocalDeform(p, Pa, Pa_, args) * l[i] / total_length

    return max(d_l, d_a) # max between access and local

def DistEuclidiana(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Which sells the line pass through and what is the size of the line that passed?

def CelulasAfetadas(A, B, args):

    # Always consistent (up to down or left to right per example)
    if B[1] < A[1] or (B[1] == A[1] and B[0] < A[0]):
        B, A = A, B

    # deltas
    d0 = B[0] - A[0]
    d1 = B[1] - A[1]

    dx = 1 if d0 >= 0 else -1
    dy = 1 if d1 >= 0 else -1

    # descritores da linha
    def fy(x):
        return d1 * 1.0 / d0 * (x - B[0]) + B[1]

    def fx(y):
        return d0 * 1.0 / d1 * (y - B[1]) + B[0]

    # indices do grid
    start_i = math.floor(A[0] / args.th)
    end_i = math.floor(B[0] / args.th)
    start_j = math.floor(A[1] / args.tw)
    end_j = math.floor(B[1] / args.tw)

    if dy < 0:
        ly = start_j * args.tw
    else:
        ly = start_j * args.tw + args.tw
    if dx < 0:
        lx = start_i * args.th
    else:
        lx = start_i * args.th + args.th

    res = [(start_i, start_j)]
    cur = A
    l = []

    while abs(start_i) != abs(end_i) or abs(start_j) != abs(end_j):

        if d0 == 0:
            t0 = np.inf
        else:
            t0 = fy(lx)
        if d1 == 0:
            t1 = np.inf
        else:
            t1 = fx(ly)

        if t0 - ly >= 0 and dx * (t1 - lx) <= 0:
            start_j += dy
            l.append(DistEuclidiana((t1, ly), cur))

            cur = t1, ly
            ly += args.tw * dy
            lasty = ly

        elif t0 - ly < 0 and dx * (t1 - lx) > 0:
            start_i += dx
            l.append(DistEuclidiana((lx, t0), cur))
            cur = lx, t0
            lx += args.th * dx

        else:
            start_i += dx
            start_j += dy
            l.append(DistEuclidiana((lx, ly), cur))

            cur = lx, ly
            lx += args.th * dx
            ly += args.tw * dy

        res.append((start_i, start_j))

    l.append(DistEuclidiana(B, cur))

    new_l = []
    new_res = []
    for i, val in enumerate(l):
        if val > 1:
            new_l.append(val)
            new_res.append(res[i])

    return new_res, new_l

# utilizar um hash map (dicionario) para criar um grafo guardar as linhas de forma mais pratica
# ideia não explicita no artigo
def HashMapLines(polylines):

    connects = {}
    polylines_hash = {}

    for index, seg in enumerate(polylines):
        # We iterate through every point in the polyline
        for i, point_array in enumerate(seg):

            t = tuple(point_array)

            if t not in polylines_hash: # where is the points? set
                polylines_hash[t] = set()
            polylines_hash[t].add((index, i))

            # connections set
            if i > 0:
                prev = tuple(seg[i-1])

                # Connect Current -> Previous
                if t not in connects: connects[t] = set()
                connects[t].add(prev)

                # Connect Previous -> Current
                if prev not in connects: connects[prev] = set()
                connects[prev].add(t)

    return connects, polylines_hash

def Optimize(Rh, polylines, polylines_orig_arg, target_W, target_H, letters, args):

    # Local Deformation Constraint
    # Over a line segment

    Rw = args.cols

    polylines_dict, polylines_hash = HashMapLines(polylines)
    aiss = np.zeros((Rh, Rw), dtype=float)
    final = np.zeros((Rh, Rw), dtype=str)

    c=0

    rad = min(args.tw, args.th)
    raster = asciiV.RasterizeLines(polylines, (target_H, target_W), 1)
    h, w = raster.shape

    # Perform a initial classification to start
    for j in range(Rh):
        sen = ''
        for i in range(Rw):
            # get cell to classify
            img_data = raster[j * args.th : j * args.th + args.th,
                              i * args.tw:i * args.tw + args.tw]
            if np.sum(img_data) == 0:
                continue

            min_cost, let = asciiA.GetAISSChar(img_data, letters, args)
            aiss[j, i] = min_cost
            final[j, i] = let

            chara = let if ord(let) > 0 else ' '

    initial_aiss = np.sum(aiss) # why sum?
    count_nz = np.count_nonzero(aiss) # good --> K
    if count_nz == 0:
        E = 0 # If empty image
    else:
        E = initial_aiss/count_nz # initial energy without deform
        # 1/K * SUM(AISS * DEFORM) --> Defrom = 0 para inicial

    if args.limit >= 0:
        progress = tqdm(total=args.limit)
    else:
        progress = tqdm(total=args.reject)

    flag = True
    not_changed = 0
    while flag:

        if args.limit < 0 and not_changed > args.reject:
            flag = False
        if args.limit >= 0 and c >= args.limit:
            flag = False

        D_cell = np.zeros((Rh, Rw), dtype=float) # deformação para celula

        # We save everything because if the move is bad, we must revert.
        back_up_raster = copy.deepcopy(raster)
        back_up_aiss = copy.deepcopy(aiss)
        back_up_final = copy.deepcopy(final)
        back_up_polylines_dict = copy.deepcopy(polylines_dict)
        back_up_polylines_hash = copy.deepcopy(polylines_hash)
        back_up_polylines = copy.deepcopy(polylines)

        point = random.choice(list(polylines_dict.keys()))
        near_points = polylines_dict[point]

        dx = np.random.uniform(-max(args.th, args.tw), max(args.th, args.tw) + 1)
        dy = np.random.uniform(-max(args.th, args.tw), max(args.th, args.tw) + 1)
        x = int(dx + point[0])
        y = int(dy + point[1])
        x = w-1 if x > w-1 else x if x > 0 else 0
        y = h-1 if y > h-1 else y if y > 0 else 0

        new_point = (x, y) # mudança diferente, movendo de forma radial

        # Update Graph Topology (Move point in dict)
        if new_point in polylines_dict:
            polylines_dict[new_point] = polylines_dict[new_point] | polylines_dict.pop(point)
            # se se mover pra cima de um ponto eles se juntam (| é união de sets)
        else:
            polylines_dict[new_point] = polylines_dict.pop(point)
            # se não só pega as conexões do ponto anterior e coloca no novo ponto

        raster = raster
        # atualiza o grid the pixels
        for p in near_points:
            polylines_dict[p].remove(point)
            polylines_dict[p].add(new_point)
            rr, cc = line(point[0], point[1], p[0], p[1])
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            raster[rr[valid], cc[valid]] = False

            rr, cc = line(new_point[0], new_point[1], p[0], p[1])
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            raster[rr[valid], cc[valid]] = True


        # calcula posição da celula escolhida no grid
        cur_chosen_cell = (math.floor(point[0] / args.th), math.floor(point[1] / args.tw))
        cur_chosen_cell_D = []
        computed_cells = []
        wrong_point_chosen = False

        # para cada um dos pontos próximos ao ponto mudado
        for p in near_points:
            cur_chosen_cell_l = 0
            cells, l = CelulasAfetadas(p, point, args)
            total_l = sum(l)

            new_cells, rubbish = CelulasAfetadas(p, new_point, args)

            D = Deform(polylines, p, point, new_point, args) # deformação

            if D < 0:
                wrong_point_chosen = True
                break

            Dtotal = 0

            # recompute new character cells
            for cell in new_cells:
                # recompute aiss each cell
                if cell[0] >= Rh or cell[1] >= Rw:
                    continue

                img_data = raster[int(cell[0] * args.th):int(cell[0] * args.th + args.th),
                                 int(cell[1] * args.tw):int(cell[1] * args.tw + args.tw)]

                aiss[cell], final[cell] = asciiA.GetAISSChar(img_data, letters, args)


            # compute every cell's deform
            for index, cell in enumerate(cells):

                if cell[0] >= Rh or cell[1] >= Rw:
                    continue

                # do not compute the repeated cell again
                if cell not in computed_cells:

                    if cell != cur_chosen_cell or cur_chosen_cell_l <= 0:

                        for s in polylines:

                            # Iterate through all points (line segments) in the polyline
                            for k in range(len(s) - 1):
                                p_start = s[k]
                                p_end = s[k+1]

                                # Calculate cells and lengths for this specific segment
                                t_cells, t_l = CelulasAfetadas(p_start, p_end, args)

                                for t_index, t_cell in enumerate(t_cells):
                                    if cell == t_cell:
                                        D_cell[cell] += t_l[t_index]

                        # recompute aiss for each affected cell
                        img_data = raster[int(cell[0] * args.th):int(cell[0] * args.th + args.th),
                                          int(cell[1] * args.tw):int(cell[1] * args.tw + args.tw)]

                        aiss[cell], final[cell] = asciiA.GetAISSChar(img_data, letters, args)

                    # record the chosen cell total length and partial Deform value
                    if cell == cur_chosen_cell:
                        cur_chosen_cell_l = D_cell[cell]
                        if index < len(l):
                            cur_chosen_cell_D.append((l[index], D))
                        continue

                    if index < len(l) and D_cell[cell] != 0:
                        D_cell[cell] = (D_cell[cell] - l[index] + l[index] * D) / D_cell[cell]
                        computed_cells.append(cell)

            if wrong_point_chosen:
                raster = back_up_raster
                aiss = back_up_aiss
                final = back_up_final
                polylines_dict = back_up_polylines_dict
                progress.update(1)
                continue


        K = np.count_nonzero(aiss)
        D_cell_new = np.ones((Rh, Rw), dtype=float)
        not_empty_cells = np.where(D_cell > 0)
        for i, j in zip(*not_empty_cells):
            D_cell_new[i, j] = D_cell[i, j]

        if K > 0:
            cur_E = np.sum(np.multiply(D_cell_new, aiss))/K # <---- K = non-empty cells
        else:
            cur_E = E

        diff = abs(cur_E - E)
        c += 1

        # print(f"Delta {diff} | cur_E {cur_E} | E {E} | Temp {0.2 * initial_aiss * c**0.997}")
        if cur_E < E:
            not_changed = 0
            E = cur_E

            for position in polylines_hash[point]:
                polylines[position[0]][position[1]] = np.array(new_point)
            if new_point in polylines_hash:
                polylines_hash[new_point] = polylines_hash[new_point] | polylines_hash.pop(point)
            else:
                polylines_hash[new_point] = polylines_hash.pop(point)

            if(args.limit < 0):
                progress.reset()
            else:
                progress.update(1)
            continue

        # SIMULATED ANNEALING
        not_changed += 1
        if c > 0 and initial_aiss > 0:
            Pr = math.exp(-diff / (0.2 * initial_aiss * c**0.997))
        else:
            Pr = 0

        if Pr < np.random.uniform(0.0, 1.0):
            # Accept Bad Move -> meta heuristic
            E = cur_E
            for position in polylines_hash[point]:
                polylines[position[0]][position[1]] = np.array(new_point)

            if new_point in polylines_hash:
                polylines_hash[new_point] = polylines_hash[new_point] | polylines_hash.pop(point)

            else:
                polylines_hash[new_point] = polylines_hash.pop(point)

            progress.update(1)
        else:
            # Reject
            raster = back_up_raster
            aiss = back_up_aiss
            final = back_up_final
            polylines_dict = back_up_polylines_dict
            polylines_hash = back_up_polylines_hash
            polylines = back_up_polylines
        progress.update(1)

    progress.close()
    return final
