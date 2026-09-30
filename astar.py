import heapq
import math
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import distance_transform_edt, map_coordinates


# ===========================================================================
# CONFIGURAÇÃO DO ROBÔ
# ===========================================================================

# reúne em um só lugar todas as medidas e limites físicos do robô
# (tamanho, rodas, velocidades, aceleração) e os parâmetros do controlador.
# O resto do código lê estes valores; ajuste-os para o robô real.
@dataclass
class RobotConfig:
    """Robô diferencial (2 rodas) de corpo circular."""
    resolution: float = 0.05      # metros por célula do mapa (veja o .yaml do mapa)
    radius: float = 0.07          # raio do robô (m)
    safety_margin: float = 0.02   # folga extra desejada além do raio (m)
    comfort_extra: float = 0.08   # a suavização evita se aproximar mais que raio+isso (m)
    wheel_base: float = 0.12      # distância entre as rodas (m)
    v_max: float = 4.20           # velocidade linear máxima (m/s)
    wheel_max: float = 4.25       # velocidade máxima de cada roda (m/s)
    w_max: float = 2.0            # velocidade angular máxima (rad/s)
    v_min: float = 0.03           # velocidade mínima em movimento (m/s)
    a_max: float = 0.4            # aceleração linear máxima (m/s²)
    alpha_max: float = 3.0        # aceleração angular máxima (rad/s²)
    slow_dist: float = 0.15       # reduz a velocidade quando a folga da parede é menor que isso (m)
    path_step: float = 0.05       # espaçamento entre pontos do caminho final (m)
    lookahead_min: float = 0.15   # pure pursuit: lookahead mínimo (m) - cenoura na frente do robo
    lookahead_max: float = 0.40   # pure pursuit: lookahead máximo (m)
    lookahead_gain: float = 0.6   # pure pursuit: lookahead = min + gain * v
    goal_tol: float = 0.06        # tolerância para considerar o objetivo alcançado (m)


# ===========================================================================
# PLANEJADOR DE CAMINHO (A*)
# ===========================================================================
class AStarPathfinder:
    # Vizinhança de 8 direções: (dl, dc, custo do passo)
    NEIGHBORS = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    # guarda o mapa, início, objetivo e parâmetros do robô, e já
    # calcula a distância de cada célula até a parede mais próxima e o campo
    # potencial (custo extra perto das paredes). Não planeja nada ainda.
    def __init__(self, map_array: np.array, start: tuple, goal: tuple, wall_influence=5.0, buffer_factor=2.0,
                 unknown_lookahead=None, robot: RobotConfig = None):
        """
        Inicializa o A* com mapa, ponto inicial, objetivo e parâmetros de influência.

        Args:
            map_array (np.array): Mapa binário (obstáculos e caminho livre).
            start (tuple): Ponto inicial (linha, coluna).
            goal (tuple): Ponto objetivo (linha, coluna).
            wall_influence (float): Peso da proximidade das paredes (preferência suave).
            buffer_factor (float): Escala da influência das paredes, em células.
            unknown_lookahead (int | None): Quantas células desconhecidas o caminho
                pode conter após a fronteira com a área conhecida. None mantém o
                caminho inteiro (o robô revela o desconhecido com o laser).
            robot (RobotConfig): Parâmetros físicos do robô.
        """
        self.robot = robot or RobotConfig()
        self.unknown_lookahead = unknown_lookahead
        self.start = start
        self.goal = goal
        self.wall_influence = wall_influence
        self.buffer_factor = buffer_factor
        self.GOAL_REACHEABLE = False
        self.clear_used_cells = None   # folga (em células) usada no último planejamento
        self.dense_path = None         # caminho bruto do A*
        self.path_smooth = None        # caminho suavizado
        self.speed_profile = None      # velocidade recomendada em cada ponto

        # Prepara o mapa, expandindo suas bordas e ajustando o array.
        self.map = map_array.copy()
        self.map_array = self.preprocess_map(map_array)

        # Distância (em células) até o obstáculo mais próximo + campo potencial.
        self.dist = self.compute_distance()
        self.potential_field = self.create_potential_field()
        self.allowed = None

    # padroniza o mapa em três valores: 0 = parede, 128 = desconhecido,
    # 255 = livre. Tons de cinza intermediários viram parede (mais seguro).
    def preprocess_map(self, map_array: np.array) -> np.array:
        """
        Ajusta o mapa, convertendo valores intermediários para obstáculos.

        Args:
            map_array (np.array): Mapa original.

        Returns:
            np.array: Mapa processado.
        """
        map_array = np.asarray(map_array)
        if map_array.ndim != 2:
            raise ValueError("O mapa deve ser um array bidimensional.")

        processed_map = np.full(map_array.shape, 128, dtype=np.uint8)
        processed_map[(map_array != 128) & (map_array < 250)] = 0
        processed_map[map_array >= 250] = 255

        return processed_map

    # para cada célula do mapa, calcula a distância (em células) até a
    # parede mais próxima. Essa matriz é a base tanto do bloqueio duro (raio do
    # robô) quanto do campo potencial e da suavização.
    def compute_distance(self) -> np.array:
        """Distância euclidiana (em células) de cada célula ao obstáculo mais próximo."""
        is_traversable = (self.map_array != 0)
        if is_traversable.all():
            # Sem nenhuma parede conhecida: distância "infinita" em todo lugar
            return np.full(self.map_array.shape, 1e3)
        return distance_transform_edt(is_traversable)

    # cria o custo extra de estar perto de paredes. Longe delas o custo
    # é ~1; colado nelas fica ~1 + wall_influence. O A* usa isso para preferir
    # caminhos pelo meio dos corredores (preferência suave, não proibição).
    def create_potential_field(self) -> np.array:
        """
        Gera campo potencial com base na distância de obstáculos (preferência suave).

        Returns:
            np.array: Campo potencial.
        """
        return 1.0 + self.wall_influence * np.exp(-self.dist / self.buffer_factor)

    # ------------------------------------------------------------------
    # Restrição dura: o CENTRO do robô só pode ocupar células com folga
    # ------------------------------------------------------------------

    # converte "raio do robô + margem" (metros) na distância mínima, em
    # células, que o centro do robô deve manter da parede. O +0.5 é a meia célula
    # da própria parede.
    def clearance_cells(self, margin: float) -> float:
        """
        Distância mínima (em células, centro a centro) entre o centro do robô e
        uma célula de parede para que o corpo não encoste nela. O +0.5 é a meia
        célula da própria parede.
        """
        return (self.robot.radius + margin) / self.robot.resolution + 0.5

    # monta a máscara de células onde o centro do robô pode estar:
    # não é parede e está a pelo menos `clear` células de qualquer parede.
    # Também libera uma "zona de escape" ao redor do início e do objetivo, para o
    # robô conseguir sair (ou chegar) mesmo se estiver perto de uma parede.
    def _build_allowed(self, clear: float) -> np.array:
        """Máscara de células permitidas para o centro do robô."""
        allowed = (self.map_array != 0) & (self.dist >= clear)
        h, w = allowed.shape
        # Zonas de escape: se início/objetivo já estão perto da parede, permite
        # sair (ou chegar) afastando-se dela.
        for p in (self.start, self.goal):
            r0, c0 = int(p[0]), int(p[1])
            if not (0 <= r0 < h and 0 <= c0 < w) or self.map_array[r0, c0] == 0:
                continue
            R = int(math.ceil(2 * clear)) + 1
            rs, re = max(r0 - R, 0), min(r0 + R + 1, h)
            cs, ce = max(c0 - R, 0), min(c0 + R + 1, w)
            yy, xx = np.ogrid[rs:re, cs:ce]
            circle = (yy - r0) ** 2 + (xx - c0) ** 2 <= R * R
            # dentro do círculo, aceita células que não estejam mais perto da
            # parede do que o ponto de partida/chegada já está
            zone = (self.map_array[rs:re, cs:ce] != 0) & \
                   (self.dist[rs:re, cs:ce] >= min(self.dist[r0, c0], clear))
            allowed[rs:re, cs:ce] |= zone & circle
        return allowed

    # estima o custo mínimo restante até o objetivo usando a distância
    # octil (andar em 8 direções). Guia o A* na direção certa sem superestimar o
    # custo, o que garante caminho ótimo.
    def heuristic(self, a: tuple, b: tuple) -> float:
        """
        Calcula a heurística entre dois pontos (distância octil).

        O custo mínimo por célula é 1.0 (o campo potencial nunca é menor que 1),
        portanto a distância octil é admissível e consistente.
        """
        dl = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return (dl + dc) + (math.sqrt(2) - 2) * min(dl, dc)

    # responde "o centro do robô pode ficar nesta célula?" (dentro do
    # mapa e na máscara de células permitidas).
    def _is_free(self, cell: tuple) -> bool:
        """Célula dentro do mapa e permitida para o centro do robô."""
        l, c = cell
        h, w = self.allowed.shape
        return 0 <= l < h and 0 <= c < w and self.allowed[l, c]

    # o A* propriamente dito. Explora a grade em 8 direções, sempre
    # expandindo o nó de menor (custo até aqui + estimativa até o objetivo),
    # usando só células da máscara `allowed`. Devolve os predecessores de cada nó.
    def _astar(self):
        """A* sobre a máscara `self.allowed`."""
        start = tuple(int(v) for v in self.start)
        goal = tuple(int(v) for v in self.goal)

        if not self._is_free(start) or not self._is_free(goal):
            return None, None

        counter = 0  # desempate estável no heap
        open_heap = [(self.heuristic(start, goal), counter, start)]
        came_from = {}
        g_score = {start: 0.0}   # melhor custo conhecido do início até cada célula
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return came_from, current

            for dl, dc, step in self.NEIGHBORS:
                neighbor = (current[0] + dl, current[1] + dc)
                if neighbor in closed or not self._is_free(neighbor):
                    continue

                # Impede "cortar quina" na diagonal entre dois obstáculos
                if dl != 0 and dc != 0:
                    if (not self._is_free((current[0] + dl, current[1]))
                            or not self._is_free((current[0], current[1] + dc))):
                        continue

                # custo do passo = comprimento do passo * custo do campo potencial
                tentative_g = g_score[current] + step * self.potential_field[neighbor]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (tentative_g + self.heuristic(neighbor, goal),
                                               counter, neighbor))
        return None, None

    # chama o A* respeitando o raio do robô. Começa com a folga
    # desejada (raio + margem); se não houver caminho (passagem estreita), tenta
    # de novo com margens menores, sem nunca ir abaixo do raio físico.
    def find_path(self):
        """
        Executa o A* respeitando o raio do robô.

        Tenta primeiro a folga desejada (raio + margem). Se não houver caminho
        (passagens estreitas), reduz a margem gradualmente, sem nunca ir abaixo
        do raio físico do robô.

        Returns:
            dict: Predecessores dos nós no caminho, ou None.
            tuple: O ponto final (objetivo) ou None se não encontrado.
        """
        margins = [self.robot.safety_margin, self.robot.safety_margin / 2,
                   self.robot.safety_margin / 4]
        for i, margin in enumerate(margins):
            clear = self.clearance_cells(margin)
            self.allowed = self._build_allowed(clear)
            self.clear_used_cells = clear
            came_from, final = self._astar()
            if final is not None:
                self.GOAL_REACHEABLE = True
                if i > 0:
                    print(f"Aviso: passagem estreita; margem reduzida para {margin * 100:.1f} cm.")
                return came_from, final

        print("Caminho não encontrado")
        return None, None

    # transforma o dicionário de predecessores do A* numa lista de
    # células ordenada do início ao objetivo (segue os predecessores de trás
    # para frente e inverte a lista).
    def reconstruct_path(self, came_from: dict, current: tuple) -> list:
        """Reconstrói o caminho a partir do ponto final até o inicial."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    #  decide o que fazer com o trecho do caminho que passa por área
    # desconhecida. Por padrão mantém tudo (o laser revela o mapa enquanto o
    # robô anda). Com `unknown_lookahead`, corta o caminho depois de N células
    # desconhecidas para replanejar quando o mapa for atualizado.
    def know_path(self, path: list) -> list:
        """
        Ajusta o caminho em relação à área desconhecida.

        O robô anda no desconhecido (o laser o revela conforme ele avança),
        então por padrão o caminho é mantido inteiro. Se `unknown_lookahead`
        for um inteiro, o caminho é cortado após essa quantidade de células
        desconhecidas a partir da fronteira, para replanejar depois.
        """
        if self.unknown_lookahead is not None:
            adjusted = []
            unknown_count = 0
            for cell in path:
                if self.map_array[cell[0], cell[1]] == 128:
                    if unknown_count >= self.unknown_lookahead:
                        break
                    unknown_count += 1
                adjusted.append(cell)
            path = adjusted

        # só é "alcançável" se o caminho (possivelmente cortado) chega ao objetivo
        self.GOAL_REACHEABLE = bool(path) and tuple(path[-1]) == tuple(self.goal)
        if not self.GOAL_REACHEABLE:
            print("Aviso: caminho truncado; o objetivo será alcançado após replanejar.")
        return path

    # reduz o caminho a poucos pontos, mantendo só onde a direção muda.
    # Hoje é usado apenas se você quiser um caminho "poligonal" simples; o fluxo
    # principal usa o caminho suavizado.
    def simplify_path(self, path: list) -> list:
        """Simplifica o caminho removendo direções repetidas."""
        if not path or len(path) < 3:
            return list(path) if path else []

        simplified = [path[0]]
        prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])

        for i in range(1, len(path) - 1):
            new_dir = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if new_dir != prev_dir:
                simplified.append(path[i])
                prev_dir = new_dir

        simplified.append(path[-1])
        return simplified

    # ------------------------------------------------------------------
    # Suavização, curvatura e perfil de velocidade
    # ------------------------------------------------------------------

    # dá a distância até a parede em posições contínuas (não só nas
    # células inteiras), por interpolação bilinear da matriz de distâncias.
    def clearance_at(self, pts: np.array) -> np.array:
        """Distância (em células) até a parede em pontos contínuos (linha, coluna)."""
        pts = np.asarray(pts, dtype=float)
        return map_coordinates(self.dist, [pts[:, 0], pts[:, 1]], order=1, mode='nearest')

    # redistribui os pontos de um caminho para ficarem igualmente
    # espaçados ao longo do comprimento (pontos a cada `step` células).
    @staticmethod
    def resample(pts: np.array, step: float) -> np.array:
        """Reamostra a polilinha com espaçamento uniforme (comprimento de arco)."""
        pts = np.asarray(pts, dtype=float)
        if len(pts) < 2:
            return pts
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])   # distância acumulada
        if s[-1] < 1e-9:
            return pts[:1]
        n = max(int(math.ceil(s[-1] / step)), 1) + 1
        ss = np.linspace(0, s[-1], n)
        return np.column_stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])])

    #  tira as "quinas" do caminho do A* (que só vira em múltiplos de
    # 45°). Cada ponto é puxado para a média dos vizinhos (deixa curvo) e um
    # pouco para onde estava (não se afasta demais). Um ponto só se move se não
    # ficar mais perto da parede do que já estava, então a suavização nunca
    # piora a folga.
    def smooth_path(self, path: list, alpha=0.005, beta=0.25, iterations=1500) -> np.array:
        """
        Suaviza o caminho do A* (que só vira em múltiplos de 45°).

        Cada ponto é puxado para a média dos vizinhos (suavidade) e levemente
        para a posição original (fidelidade). Um ponto só se move se continuar
        pelo menos tão longe da parede quanto min(sua folga original, folga de
        conforto): a suavização nunca aproxima o robô da parede.

        Args:
            path (list): Caminho denso do A* (linha, coluna).
            alpha (float): Peso de fidelidade ao caminho original.
            beta (float): Peso de suavização.
            iterations (int): Máximo de iterações.

        Returns:
            np.array: Caminho suave (N, 2) em células, espaçado uniformemente.
        """
        rb = self.robot
        step = rb.path_step / rb.resolution
        P0 = self.resample(np.asarray(path, dtype=float), step)
        if len(P0) < 3:
            return P0

        # folga mínima que cada ponto deve manter: a que já tinha (limitada ao "conforto")
        comfort = (rb.radius + rb.comfort_extra) / rb.resolution + 0.5
        thresh = np.minimum(self.clearance_at(P0), comfort) - 1e-6

        P = P0.copy()
        for _ in range(iterations):
            # nova posição proposta para os pontos internos (início e fim ficam fixos)
            new = P[1:-1] + alpha * (P0[1:-1] - P[1:-1]) + beta * (P[:-2] + P[2:] - 2 * P[1:-1])
            # aceita o movimento só onde a folga continua suficiente
            ok = self.clearance_at(new) >= thresh[1:-1]
            move = np.where(ok[:, None], new - P[1:-1], 0.0)
            P[1:-1] += move
            if np.abs(move).max() < 1e-4:   # convergiu
                break

        return self.resample(P, step)

    # mede o quanto o caminho "entorta" em cada ponto (curvatura, em
    # 1/metro), usando o círculo que passa por 3 pontos consecutivos. Curvatura
    # alta = curva fechada = robô precisa ir mais devagar.
    def curvature(self, pts: np.array) -> np.array:
        """Curvatura com sinal (1/m) em cada ponto, pelo círculo de 3 pontos."""
        X = np.asarray(pts, dtype=float) * self.robot.resolution
        k = np.zeros(len(X))
        if len(X) < 3:
            return k
        a = X[1:-1] - X[:-2]
        b = X[2:] - X[1:-1]
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        lc = np.linalg.norm(X[2:] - X[:-2], axis=1)
        k[1:-1] = 2 * cross / (la * lb * lc + 1e-12)
        return k

    # calcula a velocidade linear recomendada em cada ponto do
    # caminho. Reduz a velocidade em curvas fechadas (limites de ω e das rodas)
    # e perto de paredes, e aplica rampas para acelerar suavemente na partida e
    # frear até parar no objetivo.
    def compute_speed_profile(self, pts: np.array) -> np.array:
        """
        Velocidade linear (m/s) recomendada em cada ponto do caminho.

        Considera: curvatura (limite de ω e de velocidade de cada roda),
        proximidade das paredes (anda devagar em passagens estreitas) e
        rampas de aceleração/frenagem (parte e termina parado).
        """
        rb = self.robot
        n = len(pts)
        if n < 2:
            return np.zeros(n)
        X = np.asarray(pts, dtype=float) * rb.resolution
        ds = np.linalg.norm(np.diff(X, axis=0), axis=1)   # distância entre pontos (m)
        kap = np.abs(self.curvature(pts))

        # limite por curvatura: ω = v·κ não pode passar de w_max, e a roda
        # externa (v·(1+κL/2)) não pode passar de wheel_max
        v = np.full(n, rb.v_max)
        nz = kap > 1e-6
        v[nz] = np.minimum(v[nz], rb.w_max / kap[nz])
        v = np.minimum(v, rb.wheel_max / (1.0 + kap * rb.wheel_base / 2.0))

        # limite por proximidade da parede: quanto menos folga, mais devagar
        free_m = (self.clearance_at(pts) - 0.5) * rb.resolution - rb.radius
        v *= np.clip(free_m / rb.slow_dist, 0.3, 1.0)
        v = np.maximum(v, rb.v_min)

        # rampas: parte e termina parado; nunca acelera/freia mais que a_max
        v[0] = 0.0
        v[-1] = 0.0
        for i in range(n - 1):            # passada para frente (aceleração)
            v[i + 1] = min(v[i + 1], math.sqrt(v[i] ** 2 + 2 * rb.a_max * ds[i]))
        for i in range(n - 2, -1, -1):    # passada para trás (frenagem)
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * rb.a_max * ds[i]))
        return v

    # O que faz: executa o planejamento completo sem desenhar nada: A* com raio do
    # robô -> trata área desconhecida -> suaviza -> calcula velocidades. Guarda
    # os resultados nos atributos e devolve o caminho suavizado.
    def plan(self):
        """
        Planeja sem plotar: A* com folga do robô + suavização + perfil de velocidade.

        Returns:
            np.array | None: Caminho suave (N, 2) em células, ou None.
        """
        came_from, final_node = self.find_path()
        if final_node is None:
            return None
        dense = self.know_path(self.reconstruct_path(came_from, final_node))
        if len(dense) < 2:
            return None
        self.dense_path = dense
        self.path_smooth = self.smooth_path(dense)
        self.speed_profile = self.compute_speed_profile(self.path_smooth)
        return self.path_smooth

    # resume a qualidade do caminho suavizado: menor folga até a
    # parede, curvatura máxima, maior giro entre pontos e comprimento total.
    # Serve para conferir se o caminho é seguro e executável.
    def path_report(self) -> dict:
        """Métricas do caminho suave: folga mínima, curvatura, mudança de direção."""
        P = self.path_smooth
        rb = self.robot
        free = (self.clearance_at(P) - 0.5) * rb.resolution
        kap = np.abs(self.curvature(P))
        head = np.unwrap(np.arctan2(np.diff(P[:, 0]), np.diff(P[:, 1])))
        return {
            "folga_min_parede_cm": float(free.min() * 100),
            "raio_do_robo_cm": rb.radius * 100,
            "curvatura_max_1_m": float(kap.max()),
            "maior_giro_entre_pontos_graus": float(np.degrees(np.abs(np.diff(head)).max())) if len(head) > 1 else 0.0,
            "comprimento_m": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() * rb.resolution),
        }

    # desenha o mapa com início, objetivo, o caminho bruto do A*
    # (magenta) e o caminho suavizado (vermelho).
    def plot_path(self, path: list, smooth_path):
        """Exibe o mapa com o caminho do A* e o caminho suavizado."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.map, cmap='gray')
        plt.scatter(self.start[1], self.start[0], color='green', s=100, label='Início')
        plt.scatter(self.goal[1], self.goal[0], color='blue', s=100, label='Objetivo')

        if path is not None and len(path) > 0:
            pr, pc = zip(*path)
            plt.plot(pc, pr, color='magenta', linewidth=1, label='A* (bruto)')
            sm = np.asarray(smooth_path)
            plt.plot(sm[:, 1], sm[:, 0], color='red', linewidth=2, label='Caminho suavizado')
        else:
            plt.title("Caminho não encontrado")

        plt.legend()
        plt.axis('equal')
        plt.show()

    # é o ponto de entrada para uso com um mapa já conhecido:
    # planeja, imprime as métricas, opcionalmente plota, e devolve o caminho
    # suavizado como lista de (linha, coluna) em células.
    def run(self, show_path=True):
        """
        Executa o processo completo: busca com raio do robô, suavização e perfil de velocidade.

        Args:
            show_path (bool): Se True, exibe o caminho graficamente.

        Returns:
            list or None: Caminho suave [(linha, coluna), ...] em células (float),
                espaçado por `robot.path_step`. Perfil de velocidade em `self.speed_profile`.
        """
        print("Iniciando busca pelo caminho...")
        smooth = self.plan()
        if smooth is None:
            print("Nenhum caminho pôde ser encontrado.")
            return None

        rep = self.path_report()
        print("Métricas: " + ", ".join(f"{k}={v:.2f}" for k, v in rep.items()))
        if show_path:
            self.plot_path(self.dense_path, smooth)
        return [tuple(p) for p in smooth]


# ===========================================================================
# LEITURA DO MAPA
# ===========================================================================

# lê o arquivo PGM do mapa e converte para 0 (parede), 128
# (desconhecido) e 255 (livre); inverte o eixo vertical para alinhar com o
# sistema de coordenadas do mapa; e adiciona uma borda desconhecida de 200
# células (embaixo e à direita) para o robô poder explorar além do mapa.
def prep_map(map_path: str) -> np.array:
    """
    Prepara o mapa carregando e processando a imagem de entrada.

    Args:
        map_path (str): O caminho do arquivo do mapa.

    Returns:
        np.array: O mapa processado como um array numpy.
    """
    map_array = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if map_array is None:
        raise FileNotFoundError(f"Não foi possível carregar o mapa: {map_path}")

    prepared_map = np.full(map_array.shape, 128, dtype=np.uint8)
    prepared_map[(map_array != 205) & (map_array < 250)] = 0
    prepared_map[map_array >= 250] = 255

    prepared_map = np.flipud(prepared_map)
    prepared_map = np.pad(
        prepared_map,
        ((0, 200), (0, 200)),
        mode='constant',
        constant_values=128,
    )
    return prepared_map


# ===========================================================================
# SIMULAÇÃO: laser + robô diferencial seguindo o caminho (pure pursuit)
# ===========================================================================

# simula um sensor laser 360°. Lança raios a partir do robô no mapa
# real (truth); as células que o raio atravessa viram livres, onde ele bate vira
# parede, e o que está atrás da parede continua desconhecido. Atualiza `known`
# (o mapa que o robô conhece) diretamente.
def laser_scan(truth: np.array, known: np.array, pos: tuple, max_range=50, n_rays=360) -> None:
    """
    Simula um LIDAR 360° (ray casting no mapa real) e atualiza `known` in-place.

    Células atravessadas viram livres (255); onde o raio bate vira parede (0);
    o que está atrás da parede continua desconhecido (128).
    """
    h, w = truth.shape
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ds = np.arange(0, max_range, 0.5)   # amostras ao longo de cada raio

    # coordenadas (linha, coluna) de todas as amostras de todos os raios
    rr = np.rint(pos[0] + np.sin(angles)[:, None] * ds[None, :]).astype(int)
    cc = np.rint(pos[1] + np.cos(angles)[:, None] * ds[None, :]).astype(int)

    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    rc = np.clip(rr, 0, h - 1)
    ccl = np.clip(cc, 0, w - 1)
    # o raio para na primeira parede ou ao sair do mapa
    blocked = (~inside) | (truth[rc, ccl] == 0)

    has_hit = blocked.any(axis=1)
    first_hit = np.where(has_hit, blocked.argmax(axis=1), len(ds))

    idx = np.arange(len(ds))[None, :]
    # antes da parede: livre; na parede: obstáculo; depois: continua desconhecido
    free_mask = (idx < first_hit[:, None]) & inside
    known[rc[free_mask], ccl[free_mask]] = 255

    wall_mask = (idx == first_hit[:, None]) & inside & has_hit[:, None]
    known[rc[wall_mask], ccl[wall_mask]] = 0


#  gera um "mapa real" escondido do robô, só para testar o laser.
# Mantém o que o mapa PGM já conhece, considera o resto livre e espalha paredes
# aleatórias na área desconhecida. Só aceita mundos em que o objetivo é
# alcançável por um robô com o raio configurado.
def make_synthetic_world(prepared_map: np.array, start: tuple, goal: tuple,
                         n_walls=10, seed=0, robot: RobotConfig = None) -> np.array:
    """
    Cria um "mapa real" para simular o laser: mantém o que já é conhecido,
    trata o desconhecido como livre e espalha paredes aleatórias nele.
    Só aceita mundos em que o objetivo é alcançável pelo robô (com seu raio).
    """
    base = np.where(prepared_map == 0, 0, 255).astype(np.uint8)
    unknown = (prepared_map == 128)
    h, w = base.shape

    # tenta seeds diferentes até achar um mundo em que o objetivo é alcançável
    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        truth = base.copy()
        for _ in range(n_walls):
            length = int(rng.integers(25, 70))
            thick = 3
            r = int(rng.integers(0, h - 1))
            c = int(rng.integers(0, w - 1))
            # parede horizontal ou vertical, só onde o mapa era desconhecido
            if rng.random() < 0.5:
                sl = (slice(r, min(r + thick, h)), slice(c, min(c + length, w)))
            else:
                sl = (slice(r, min(r + length, h)), slice(c, min(c + thick, w)))
            region = np.zeros_like(unknown)
            region[sl] = True
            truth[region & unknown] = 0

        # garante área livre ao redor do início e do objetivo
        for p in (start, goal):
            win = (slice(max(p[0] - 6, 0), p[0] + 7), slice(max(p[1] - 6, 0), p[1] + 7))
            truth[win][unknown[win]] = 255

        # bordas do mapa são parede
        truth[0, :] = truth[-1, :] = 0
        truth[:, 0] = truth[:, -1] = 0

        test = AStarPathfinder(truth, start, goal, wall_influence=0.0, robot=robot)
        _, final = test.find_path()
        if final is not None:
            return truth
    raise RuntimeError("Não consegui gerar um mundo sintético com o objetivo alcançável.")


# normaliza um ângulo para o intervalo [-π, π], para que a diferença
# entre duas orientações seja sempre o menor giro (ex.: 350° vira -10°).
def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# roda a simulação completa. A cada passo de tempo: o laser atualiza
# o mapa; o A* replaneja se surgir parede no caminho (ou periodicamente); o
# controlador pure pursuit calcula as velocidades do robô (com limites de
# aceleração); e o robô se move. Ao final devolve as métricas (tempo, colisões,
# folga mínima real).
def simulate_exploration(truth: np.array, start: tuple, goal: tuple, robot: RobotConfig = None,
                         max_range=50, wall_influence=10.0, buffer_factor=3.0,
                         dt=0.1, replan_every=3.0, check_ahead=1.0, max_time=400.0,
                         theta0=0.0, animate=True, draw_every=5, save_path=None):
    """
    Simula o robô diferencial: o laser atualiza o mapa a cada passo, o A* replaneja
    (por paredes novas no caminho ou periodicamente) e o controlador pure pursuit
    gera velocidades de roda respeitando limites de velocidade e aceleração.

    Args:
        truth (np.array): Mapa real (0 = parede, 255 = livre).
        start (tuple): Posição inicial (linha, coluna).
        goal (tuple): Objetivo (linha, coluna).
        robot (RobotConfig): Parâmetros do robô.
        max_range (float): Alcance do laser em células.
        wall_influence, buffer_factor: Parâmetros do campo potencial.
        dt (float): Passo de simulação (s).
        replan_every (float): Replaneja a cada N segundos.
        check_ahead (float): Distância à frente (m) verificada por paredes novas.
        max_time (float): Tempo máximo (s).
        theta0 (float): Orientação inicial (rad, 0 = aponta para +coluna).
        animate (bool): Mostra a animação.
        draw_every (int): Redesenha a cada N passos.
        save_path (str | None): Salva a figura final do mapa.

    Returns:
        dict: Métricas e logs da simulação.
    """
    rb = robot or RobotConfig()
    res = rb.resolution
    known = np.full(truth.shape, 128, dtype=np.uint8)   # o robô começa sem conhecer nada
    truth_dist = distance_transform_edt(truth != 0)     # para medir colisão no mapa real
    goal = tuple(goal)
    gx, gy = goal[1] * res, goal[0] * res

    # pose do robô em metros: x = coluna*res, y = linha*res, th = orientação
    x, y, th = start[1] * res, start[0] * res, theta0
    v = w = 0.0                                          # velocidades atuais (linear, angular)
    laser_scan(truth, known, (int(round(y / res)), int(round(x / res))), max_range)

    # --- preparação da figura ---
    fig, ax = plt.subplots(figsize=(10, 10))
    img = ax.imshow(known, cmap='gray', vmin=0, vmax=255)
    ax.scatter(start[1], start[0], color='green', s=80, label='Início')
    ax.scatter(goal[1], goal[0], color='blue', s=80, label='Objetivo')
    plan_line, = ax.plot([], [], color='red', linewidth=1.5, linestyle='--', label='Caminho planejado')
    traj_line, = ax.plot([], [], color='magenta', linewidth=1.5, label='Percorrido')
    body = Circle((start[1], start[0]), rb.radius / res, color='orange', label='Robô')
    ax.add_patch(body)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    if animate:
        plt.ion()

    traj = [(y / res, x / res)]
    log = {"t": [], "vl": [], "vr": [], "v": [], "w": [], "clear_cm": []}
    plan_pts = plan_xy = v_prof = seg_s = None
    clear_used = None
    last_idx, replans, collisions, in_collision = 0, 0, 0, False
    t, t_replan = 0.0, -1e9
    status = "Tempo máximo atingido"

    # O que faz: atualiza a imagem (mapa conhecido, caminho planejado,
    # trajetória percorrida e posição do robô) e o título da janela.
    def redraw(title):
        img.set_data(known)
        if plan_pts is not None:
            plan_line.set_data(plan_pts[last_idx:, 1], plan_pts[last_idx:, 0])
        tr = np.array(traj)
        traj_line.set_data(tr[:, 1], tr[:, 0])
        body.center = (x / res, y / res)
        ax.set_title(title)
        if animate:
            plt.pause(0.001)

    # O que faz: verifica se o laser descobriu uma parede perto do trecho do
    # caminho logo à frente do robô. Se sim, o caminho atual ficou inseguro e
    # é preciso replanejar. Usa só uma janela do mapa ao redor do robô (rápido).
    def blocked_ahead(row, col):
        """Há parede recém-descoberta perto do caminho à frente?"""
        n_ahead = int(check_ahead / rb.path_step) + 1
        pts = plan_pts[last_idx:last_idx + n_ahead]
        R = int(check_ahead / res + clear_used) + 6
        r0, r1 = max(int(row) - R, 0), min(int(row) + R + 1, known.shape[0])
        c0, c1 = max(int(col) - R, 0), min(int(col) + R + 1, known.shape[1])
        crop = known[r0:r1, c0:c1]
        if crop.min() != 0:          # sem paredes conhecidas na janela
            return False
        d = distance_transform_edt(crop != 0)
        vals = map_coordinates(d, [pts[:, 0] - r0, pts[:, 1] - c0], order=1, mode='nearest')
        return bool((vals < clear_used - 0.3).any())

    while t < max_time:
        row, col = y / res, x / res
        if math.hypot(x - gx, y - gy) <= rb.goal_tol:
            status = "Objetivo alcançado"
            break

        # --- replanejamento: sem caminho, tempo esgotado ou parede à frente ---
        need = plan_pts is None or (t - t_replan) >= replan_every or blocked_ahead(row, col)
        if need:
            planner = AStarPathfinder(known, (int(round(row)), int(round(col))), goal,
                                      wall_influence, buffer_factor, robot=rb)
            sm = planner.plan()
            if sm is None:
                status = "Sem caminho possível com o mapa conhecido"
                break
            sm[0] = (row, col)  # o caminho começa exatamente na pose do robô
            plan_pts, v_prof, clear_used = sm, planner.speed_profile, planner.clear_used_cells
            plan_xy = np.column_stack([sm[:, 1], sm[:, 0]]) * res   # (x, y) em metros
            # distância acumulada ao longo do caminho, usada para achar o ponto-alvo
            seg_s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(plan_xy, axis=0), axis=1))])
            last_idx, t_replan = 0, t
            replans += 1

        # --- pure pursuit: persegue um ponto à frente no caminho ---
        N = len(plan_xy)
        # ponto do caminho mais próximo do robô (busca só logo à frente do último)
        hi = min(last_idx + 80, N)
        d = np.linalg.norm(plan_xy[last_idx:hi] - [x, y], axis=1)
        idx = last_idx + int(d.argmin())
        last_idx = idx
        # lookahead cresce com a velocidade: mais rápido => olha mais longe
        Ld = float(np.clip(rb.lookahead_min + rb.lookahead_gain * abs(v), rb.lookahead_min, rb.lookahead_max))
        j = min(int(np.searchsorted(seg_s, seg_s[idx] + Ld)), N - 1)
        tx, ty = plan_xy[j]                                       # ponto-alvo
        alpha = _wrap(math.atan2(ty - y, tx - x) - th)            # ângulo até o alvo
        Ld_act = max(math.hypot(tx - x, ty - y), 1e-3)
        # velocidade de referência: a do perfil, um lookahead à frente
        k_ref = min(idx + int(round(Ld / rb.path_step)), N - 1)
        v_ref = max(float(v_prof[k_ref]), rb.v_min)

        if abs(alpha) > math.pi / 2:          # alvo atrás: gira no lugar
            v_t, w_t = 0.0, float(np.clip(2.0 * alpha, -rb.w_max, rb.w_max))
        else:
            kappa = 2.0 * math.sin(alpha) / Ld_act               # curvatura do arco até o alvo
            # respeita o limite de ω e da velocidade de cada roda
            v_t = min(v_ref, rb.wheel_max / (1.0 + abs(kappa) * rb.wheel_base / 2.0),
                      rb.w_max / max(abs(kappa), 1e-6))
            w_t = v_t * kappa

        # rampas de aceleração => movimentos suaves (sem trancos)
        v += float(np.clip(v_t - v, -rb.a_max * dt, rb.a_max * dt))
        w += float(np.clip(w_t - w, -rb.alpha_max * dt, rb.alpha_max * dt))
        # move o robô (modelo de robô diferencial, orientação média no passo)
        th_mid = th + 0.5 * w * dt
        x += v * math.cos(th_mid) * dt
        y += v * math.sin(th_mid) * dt
        th = _wrap(th + w * dt)
        t += dt

        # o laser vê o ambiente na nova posição
        row, col = y / res, x / res
        laser_scan(truth, known, (int(round(row)), int(round(col))), max_range)

        # --- métricas contra o mapa real (só para conferir a simulação) ---
        free_cm = (float(map_coordinates(truth_dist, [[row], [col]], order=1, mode='nearest')[0]) - 0.5) * res * 100
        hit = free_cm < rb.radius * 100 - 0.1     # corpo do robô tocou a parede?
        if hit and not in_collision:
            collisions += 1
        in_collision = hit
        traj.append((row, col))
        for k_, val in zip(("t", "vl", "vr", "v", "w", "clear_cm"),
                           (t, v - w * rb.wheel_base / 2, v + w * rb.wheel_base / 2, v, w, free_cm)):
            log[k_].append(val)

        if int(round(t / dt)) % draw_every == 0:
            redraw(f"t={t:5.1f}s | v={v:.2f} m/s | replanejamentos: {replans}")

    redraw(f"{status} | t={t:.1f}s | colisões: {collisions}")
    result = {"status": status, "tempo_s": t, "replanejamentos": replans, "colisoes": collisions,
              "folga_min_cm": min(log["clear_cm"]) if log["clear_cm"] else None,
              "v_roda_max": max(max(np.abs(log["vl"])), max(np.abs(log["vr"]))) if log["t"] else 0.0,
              "log": log, "trajetoria": traj}
    print(f"{status} em {t:.1f}s | replanejamentos: {replans} | colisões: {collisions} | "
          f"folga mínima real: {result['folga_min_cm']:.1f} cm (raio {rb.radius * 100:.0f} cm) | "
          f"vel. máx. de roda: {result['v_roda_max']:.2f} m/s")

    if save_path:
        fig.savefig(save_path, dpi=100)
    if animate:
        plt.ioff()
        plt.show()
    return result


# O que faz: ponto de entrada do programa. Define início, objetivo e as medidas
# do robô, carrega o mapa e roda a simulação com laser (ou só o planejamento
# estático, se SIMULAR = False).
def main():
    START = (60, 20)
    GOAL = (60, 120)
    SIMULAR = True

    # AJUSTE com as medidas do robô real e a resolução do seu mapa (.yaml)
    robot = RobotConfig(resolution=0.05, radius=0.07, wheel_base=0.12)

    map_array = prep_map('map5.pgm')

    if SIMULAR:
        # Mundo "real" escondido do robô. Se tiver o mapa completo do ambiente:
        # truth = np.where(prep_map('mapa_completo.pgm') == 0, 0, 255)
        truth = make_synthetic_world(map_array, START, GOAL, robot=robot)
        simulate_exploration(truth, START, GOAL, robot=robot, wall_influence=10.0, buffer_factor=3.0)
    else:
        # Planejamento estático sobre o mapa conhecido (uso no robô real)
        astar = AStarPathfinder(map_array, START, GOAL, wall_influence=10.0, buffer_factor=3.0, robot=robot)
        astar.run()


if __name__ == '__main__':
    main()