import cv2
import numpy as np
import math
import random
from tqdm import tqdm
import asciiVectorize as asciiV
import asciiAISS as asciiA


def ComputeVTheta(vec_old, vec_new):

    norm_old = np.linalg.norm(vec_old)
    norm_new = np.linalg.norm(vec_new)

    if norm_old == 0 or norm_new == 0:
        return 1.0

    cos_theta = np.dot(vec_old, vec_new) / (norm_old * norm_new)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    return math.exp((8.0 / math.pi) * theta)

def safe_exp(x):
    if x > 700:
        return float('inf')
    return math.exp(x)

def ComputeVR(vec_old, vec_new, min_dim):

    r = np.linalg.norm(vec_old)
    r_linha = np.linalg.norm(vec_new)

    epsilon = 1e-9

    lambda_2 = 2.0 / (min_dim + epsilon)
    lambda_3 = 0.5

    arg1 = lambda_2 * abs(r_linha - r)


    numerator = lambda_3 * max(r, r_linha)
    denominator = min(r, r_linha) + epsilon
    arg2 = numerator / denominator

    term1 = safe_exp(arg1)
    term2 = safe_exp(arg2)

    return max(term1, term2)

def GetIntersectionRay(ray_origin, ray_dir, seg_p1, seg_p2):

    rx, ry = ray_origin
    rdx, rdy = ray_dir
    x1, y1 = seg_p1
    x2, y2 = seg_p2

    sx, sy = x2 - x1, y2 - y1

    denom = rdx * sy - rdy * sx
    if denom == 0: # se são paralelos
        return None, None

    t_ray = ((x1 - rx) * sy - (y1 - ry) * sx) / denom
    u_seg = ((x1 - rx) * rdy - (y1 - ry) * rdx) / denom

    if t_ray > 0 and 0 <= u_seg <= 1:
        ix = rx + t_ray * rdx
        iy = ry + t_ray * rdy
        return (ix, iy), u_seg # Retorna ponto e posição relativa no segmento

    return None, None

def ComputeAccess(current_idx, point_idx, polylines_current, polylines_orig, min_dim):

    # segmento atual
    line = polylines_current[current_idx]
    p1 = line[point_idx]
    if point_idx + 1 >= len(line): # ultimo ponto
        return 1.0
    p2 = line[point_idx + 1]

    midpoint_curr =  ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)

    # segmento original correspondente
    orig_line = polylines_orig[current_idx]
    op1 = orig_line[point_idx]
    op2 = orig_line[point_idx+1]
    midpoint_orig = ((op1[0]+op2[0])/2, (op1[1]+op2[1])/2)


    total_weighted_local = 0.0
    total_weight = 0.0

    # Simular "multiple rays" para 8 direções.
    angles = np.linspace(0, 2*math.pi, 8, endpoint=False)

    for ang in angles:
        ray_dir = (math.cos(ang), math.sin(ang))

        closest = None
        poly_idx = None
        seg_idx = None
        t = None
        min_dist = float('inf')

        # ray casting
        # quem é e onde esta o melhor vizinho
        for l_idx, poly in enumerate(polylines_current):
            if l_idx == current_idx: continue # Ignora auto-intersecção com a própria polilinha

            for k in range(len(poly)-1):
                inter, t_seg = GetIntersectionRay(midpoint_curr, ray_dir, poly[k], poly[k+1])

                if inter:
                    dist = math.hypot(inter[0]-midpoint_curr[0], inter[1]-midpoint_curr[1])
                    if dist < min_dist:
                        min_dist = dist

                        closest = inter
                        poly_idx = l_idx
                        seg_idx = k
                        t = t_seg

        if closest:

            # vetor pp_i do meio atual até o ponto de impacto atual
            vec_curr = np.array(closest) - np.array(midpoint_curr)

            # recuperar o vetor original
            # Usamos os índices (poly_idx, seg_idx) e a posição relativa (u) para achar

            orig_neighbor_poly = polylines_orig[poly_idx]
            orig_neighbor_seg_p1 = orig_neighbor_poly[seg_idx]
            orig_neighbor_seg_p2 = orig_neighbor_poly[seg_idx + 1]

            # achar o ponto exato no segmento original
            orig_hit_point = (1 - t) * orig_neighbor_seg_p1 + t * orig_neighbor_seg_p2

            # vetor original
            vec_orig = orig_hit_point - midpoint_orig

            # deformação para este raio
            v_theta = ComputeVTheta(vec_orig, vec_curr)
            v_r = ComputeVR(vec_orig, vec_curr, min_dim)

            d_local = max(v_theta, v_r)

            weight = 1.0 / (min_dist + 1e-5)

            total_weighted_local += weight * d_local
            total_weight += weight

    if total_weight == 0:
        return 1.0

    return total_weighted_local / total_weight


def DeformationCost(polylines_c, polylines_o, line_idx, point_idx, min_dim):

    # custo local (arestas conectadas ao vértice movido)
    d_local = 1.0

    current_pt = polylines_c[line_idx][point_idx]
    orig_pt = polylines_o[line_idx][point_idx]

    # aresta anterior
    if point_idx > 0:
        prev_pt = polylines_c[line_idx][point_idx-1]
        prev_o = polylines_o[line_idx][point_idx-1]

        vec_cur = current_pt - prev_pt
        vec_o = orig_pt - prev_o

        v_theta = ComputeVTheta(vec_o, vec_cur)
        v_r = ComputeVR(vec_o, vec_cur, min_dim)
        d_local = max(d_local, max(v_theta, v_r))

    # aresta seguinte
    if point_idx < len(polylines_c[line_idx]) - 1:
        next_pt = polylines_c[line_idx][point_idx+1]
        next_o = polylines_o[line_idx][point_idx+1]

        vec_cur = next_pt - current_pt
        vec_o = next_o - orig_pt

        v_theta = ComputeVTheta(vec_o, vec_cur)
        v_r = ComputeVR(vec_o, vec_cur, min_dim)
        d_local = max(d_local, max(v_theta, v_r))


    d_access = ComputeAccess(line_idx, point_idx, polylines_c, polylines_o, min_dim)

    # Maior enter os custos de deformação
    d_deform = max(d_local, d_access)

    return d_deform

def Optimize(Rh, polylines, polylines_orig, target_W, target_H, letters, args):

    ta = asciiA.CalculateTa(Rh, asciiV.RasterizeLines(polylines, (target_W, target_H)), letters, args)
    temp = 0.2*ta

    not_changed = 0
    min_dim = min(args.tw, args.th)
    current_energy = ta # Aproximação inicial


    # Distância máxima de deslocamento (longer side of char)
    d_max = max(args.tw, args.th)


    if(args.limit >= 0):
        progress = tqdm(total=args.limit)
    else:
        progress = tqdm(total=args.reject)

    iteration = 0
    flag = True
    while flag:
        iteration += 1

        if args.limit < 0 and not_changed > args.reject:
            flag = False
        if args.limit >= 0 and iteration >= args.limit:
            flag = False

        # vértice aleatório
        line_idx = random.randint(0, len(polylines)-1)
        if len(polylines[line_idx]) == 0: # empty
            if(args.limit >= 0):
                progress.update(1)
            continue
        point_idx = random.randint(0, len(polylines[line_idx])-1)

        old_pos = np.copy(polylines[line_idx][point_idx])

        # deslocar aleatoriamente
        dx = random.uniform(-d_max, d_max)
        dy = random.uniform(-d_max, d_max)
        new_pos = old_pos + [dx, dy]

        polylines[line_idx][point_idx] = new_pos

        # energia (local)
        # identificar célula afetada
        c = int(new_pos[0] // args.tw)
        r = int(new_pos[1] // args.th)

        # checa limites após mudança
        if r < 0 or r >= Rh or c < 0 or c >= args.cols:
            polylines[line_idx][point_idx] = old_pos # Revert
            not_changed += 1
            progress.update(1)
            continue

        # rasterizar novamente para calcular AISS
        raster_c = asciiV.RasterizeLines(polylines, (target_H, target_W), 1)
        cell_img = raster_c[r*args.th : (r+1)*args.th,
                            c*args.tw : (c+1)*args.tw]

        cell_desc = asciiA.AISS(cell_img, args)

        # encontrar melhor caractere
        best = float('inf')
        for _, char_desc in letters.items():
            d = np.linalg.norm(cell_desc - char_desc)
            if d < best:
                best = d

        # Calcular D_deform
        d_deform = DeformationCost(polylines, polylines_orig, line_idx, point_idx, min_dim)

        # Simplificado utilizando apenas a celula atual
        E_new = best * d_deform


        # Revertemos para calcular E_old exato localmente
        polylines[line_idx][point_idx] = old_pos
        raster_old = asciiV.RasterizeLines(polylines, (target_H, target_W), 1)
        cell_old = raster_old[r*args.th : (r+1)*args.th,
                              c*args.tw : (c+1)* args.tw]

        desc_old = asciiA.AISS(cell_old, args)

        old = float('inf')
        for _, cd in letters.items():
            d = np.linalg.norm(desc_old - cd)
            if d < old:
                old = d

        # D_deform old é 1.0 se não havia deformação anterior, mas temos que calcular relativo ao original
        d_deform_old = DeformationCost(polylines, polylines_orig, line_idx, point_idx, min_dim)
        E_old = old * d_deform_old

        # Reaplica movimento
        polylines[line_idx][point_idx] = new_pos

        delta = E_new - E_old

        safe_temp = max(temp, 1e-9)
        exponent = -delta / safe_temp
        if exponent > -700 and exponent < 700:
            prob = math.exp(exponent)
        elif exponent > 700:
            prob = float('inf')
        else:
            prob = 0

        if delta < 0 or random.random() < prob:
            not_changed = 0 # restart optimization counter
            if args.limit < 0:
                progress.reset()
        else:
            polylines[line_idx][point_idx] = old_pos # Revert
            not_changed += 1
            if args.limit < 0:
                progress.update(1)

        # atualizar temperatura
        temp *= args.decay
        if args.limit >= 0:
            progress.update(1)

    progress.close()
