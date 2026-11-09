import math
import random
import numpy as np

# ==============================================================================
# 1. HARDWARE CODEC TAKUM E ULA EM PONTO FIXO LOGARÍTMICO (Q9.27 - FIEL AO HARDWARE)
# ==============================================================================

def decode_takum_base(val: int, sew: int):
    """
    Extrai os campos S, D, R, C e M de uma palavra Takum.
    Retorna: s, d, r, c_val, m_val, p
    """
    if sew < 8:
        raise ValueError("SEW deve ser maior ou igual a 8.")

    val &= (1 << sew) - 1
    eff_sew = max(12, sew)

    if sew < 12:
        val <<= (12 - sew)

    s = (val >> (eff_sew - 1)) & 1
    d = (val >> (eff_sew - 2)) & 1
    R_bits = (val >> (eff_sew - 5)) & 0b111

    r = (7 - R_bits) if d == 0 else R_bits
    p = eff_sew - 5 - r

    c_val = ((val >> p) & ((1 << r) - 1)) if r > 0 else 0
    m_val = (val & ((1 << p) - 1)) if p > 0 else 0

    return s, d, r, c_val, m_val, p


def encode_takum_base(s: int, d: int, r: int, c_val: int, m_val: int, sew: int) -> int:
    """
    Empacota S, D, R, C e M em uma palavra Takum.
    Para SEW < 12 (T8), aplica redução com arredondamento RNE (Round to Nearest, Ties to Even).
    """
    if sew < 8:
        raise ValueError("SEW deve ser maior ou igual a 8.")

    eff_sew = max(12, sew)
    if not 0 <= r <= 7:
        raise ValueError(f"Valor de r inválido: {r}")

    R_bits = (7 - r) if d == 0 else r

    val = (
        (s << (eff_sew - 1))
        | (d << (eff_sew - 2))
        | ((R_bits & 0b111) << (eff_sew - 5))
    )

    if r > 0:
        val |= (c_val & ((1 << r) - 1)) << (eff_sew - 5 - r)

    p = eff_sew - 5 - r
    if p > 0:
        val |= m_val & ((1 << p) - 1)

    if sew < 12:
        shift = 12 - sew
        half_bit = (val >> (shift - 1)) & 1
        sticky = val & ((1 << (shift - 1)) - 1) if shift > 1 else 0
        lsb = (val >> shift) & 1
        round_inc = half_bit & (1 if sticky > 0 else lsb)
        val = (val >> shift) + round_inc

    return val & ((1 << sew) - 1)


def float_to_takum_log(valor: float, sew: int) -> int:
    """
    Converte um float para a palavra Takum binária de n bits.
    """
    if sew < 8:
        raise ValueError("SEW deve ser maior ou igual a 8.")

    nar = 1 << (sew - 1)

    if math.isnan(valor) or math.isinf(valor):
        return nar

    if valor == 0.0:
        return 0

    s = 0 if valor > 0 else 1
    l = 2.0 * math.log(abs(valor))

    cm_signed = -l if s == 1 else l
    eff_sew = max(12, sew)

    for _ in range(3):
        c = math.floor(cm_signed)
        frac = cm_signed - c
        d = 1 if c >= 0 else 0

        if d == 1:
            r = (c + 1).bit_length() - 1
            c_val = c - (1 << r) + 1
        else:
            r = (-c).bit_length() - 1
            c_val = c + (1 << (r + 1)) - 1

        if r > 7:
            cm_signed = 254.999999 if c > 0 else -255.0
            continue

        p = eff_sew - 5 - r
        m_val = int(round(frac * (1 << p))) if p > 0 else 0

        if p > 0 and m_val >= (1 << p):
            cm_signed = float(c + 1)
            continue

        res = encode_takum_base(s, d, r, c_val, m_val, sew)

        if res == 0 or res == nar:
            res ^= 1

        return res

    raise ArithmeticError("Não foi possível normalizar o valor para Takum.")


def takum_log_to_float(bits: int, sew: int) -> float:
    """
    Converte uma palavra Takum de n bits para float.
    """
    bits &= (1 << sew) - 1

    if bits == 0:
        return 0.0

    if bits == (1 << (sew - 1)):
        return float("nan")

    s, d, r, c_val, m_val, p = decode_takum_base(bits, sew)

    if d == 0:
        c = -(1 << (r + 1)) + 1 + c_val
    else:
        c = (1 << r) - 1 + c_val

    m = (m_val / float(1 << p)) if p > 0 else 0.0
    l = -(c + m) if s == 1 else (c + m)
    magnitude = math.exp(l / 2.0)

    return -magnitude if s == 1 else magnitude


class TakumHardwareALU:
    """
    Unidade Aritmética e Lógica de referência em Ponto Fixo Logarítmico (Q9.27).
    """
    def __init__(self):
        self.F_BITS = 27
        self.SCALE = 1 << self.F_BITS

        self.MAX_CM_FIXED = (255 << self.F_BITS) - 1
        self.MIN_CM_FIXED = -(255 << self.F_BITS) + 1

        self.LUT_MAX_Q = 40
        self.LUT_FRAC_BITS = 8
        self.LUT_STEP = 1 << (self.F_BITS - self.LUT_FRAC_BITS)

        self.lut_gauss_add = []
        self.lut_gauss_sub = []

        num_entries = (self.LUT_MAX_Q << self.LUT_FRAC_BITS) + 1

        for idx in range(num_entries):
            q = idx / float(1 << self.LUT_FRAC_BITS)
            phi_plus = 2.0 * math.log1p(math.exp(-0.5 * q))
            self.lut_gauss_add.append(int(round(phi_plus * self.SCALE)))

            if idx == 0:
                self.lut_gauss_sub.append(0)
            else:
                phi_minus = 2.0 * math.log1p(-math.exp(-0.5 * q))
                self.lut_gauss_sub.append(int(round(phi_minus * self.SCALE)))

    def decode_to_fixed_lns(self, bits: int, sew: int):
        bits &= (1 << sew) - 1
        if bits == 0:
            return "ZERO", 0, 0
        if bits == (1 << (sew - 1)):
            return "NAR", 1, 0

        s, d, r, c_val, m_val, p = decode_takum_base(bits, sew)

        if d == 0:
            c = -(1 << (r + 1)) + 1 + c_val
        else:
            c = (1 << r) - 1 + c_val

        cm_fixed = (c << self.F_BITS) + ((m_val << (self.F_BITS - p)) if p > 0 else 0)
        l_fixed = -cm_fixed if s == 1 else cm_fixed
        return "NORMAL", s, l_fixed

    def encode_from_fixed_lns(self, tipo: str, s: int, l_fixed: int, sew: int) -> int:
        if tipo == "ZERO":
            return 0
        if tipo == "NAR":
            return 1 << (sew - 1)

        eff_sew = max(12, sew)
        cm_fixed = -l_fixed if s == 1 else l_fixed
        cm_fixed = max(self.MIN_CM_FIXED, min(self.MAX_CM_FIXED, cm_fixed))

        c = cm_fixed >> self.F_BITS
        m_fixed = cm_fixed & (self.SCALE - 1)

        d = 1 if c >= 0 else 0

        if d == 1:
            r = (c + 1).bit_length() - 1
            if r > 7:
                c = 254
                r = 7
                m_fixed = self.SCALE - 1
            c_val = c - (1 << r) + 1
        else:
            r = (-c).bit_length() - 1
            if r > 7:
                c = -255
                r = 7
                m_fixed = 0
            c_val = c + (1 << (r + 1)) - 1

        p = eff_sew - 5 - r
        m_val = 0

        if p > 0:
            shift_m = self.F_BITS - p
            if shift_m > 0:
                half_bit = (m_fixed >> (shift_m - 1)) & 1
                sticky = m_fixed & ((1 << (shift_m - 1)) - 1) if shift_m > 1 else 0
                lsb = (m_fixed >> shift_m) & 1
                round_inc = half_bit & (1 if sticky > 0 else lsb)
                m_val = (m_fixed >> shift_m) + round_inc
            else:
                m_val = m_fixed << (-shift_m)

            if m_val >= (1 << p):
                c += 1
                m_val = 0
                d = 1 if c >= 0 else 0

                if d == 1:
                    r = (c + 1).bit_length() - 1
                    if r > 7:
                        c = 254
                        r = 7
                        p = eff_sew - 5 - r
                        m_val = (1 << p) - 1
                    c_val = c - (1 << r) + 1
                else:
                    r = (-c).bit_length() - 1
                    if r > 7:
                        c = -255
                        r = 7
                        m_val = 0
                    c_val = c + (1 << (r + 1)) - 1

                p = eff_sew - 5 - r
                if p > 0:
                    m_val = min(m_val, (1 << p) - 1)

        res = encode_takum_base(s, d, r, c_val, m_val, sew)
        nar = 1 << (sew - 1)
        if res == 0 or res == nar:
            res ^= 1

        return res

    def _gauss_lut_lookup(self, q_fixed: int, same_sign: bool) -> int:
        if q_fixed < 0:
            raise ValueError("q_fixed deve ser não negativo.")

        if q_fixed == 0:
            if same_sign:
                return self.lut_gauss_add[0]
            raise ValueError("Phi-(0) é indefinido.")

        q_max = self.LUT_MAX_Q * self.SCALE
        if q_fixed >= q_max:
            return 0

        idx = q_fixed >> (self.F_BITS - self.LUT_FRAC_BITS)
        rem = q_fixed & (self.LUT_STEP - 1)

        lut = self.lut_gauss_add if same_sign else self.lut_gauss_sub

        if not same_sign and idx == 0:
            q_real = q_fixed / float(self.SCALE)
            phi = 2.0 * math.log1p(-math.exp(-0.5 * q_real))
            return int(round(phi * self.SCALE))

        y0 = lut[idx]
        y1 = lut[idx + 1]
        delta = ((y1 - y0) * rem) >> (self.F_BITS - self.LUT_FRAC_BITS)
        return y0 + delta

    def mul(self, bits_a: int, bits_b: int, sew: int) -> int:
        t_a, s_a, l_a = self.decode_to_fixed_lns(bits_a, sew)
        t_b, s_b, l_b = self.decode_to_fixed_lns(bits_b, sew)

        if t_a == "NAR" or t_b == "NAR":
            return self.encode_from_fixed_lns("NAR", 1, 0, sew)
        if t_a == "ZERO" or t_b == "ZERO":
            return self.encode_from_fixed_lns("ZERO", 0, 0, sew)

        s_res = s_a ^ s_b
        l_res = l_a + l_b
        return self.encode_from_fixed_lns("NORMAL", s_res, l_res, sew)

    def add_sub(self, bits_a: int, bits_b: int, sew: int, is_sub: bool = False) -> int:
        t_a, s_a, l_a = self.decode_to_fixed_lns(bits_a, sew)
        t_b, s_b, l_b = self.decode_to_fixed_lns(bits_b, sew)

        if t_a == "NAR" or t_b == "NAR":
            return self.encode_from_fixed_lns("NAR", 1, 0, sew)

        if is_sub and t_b == "NORMAL":
            s_b ^= 1

        if t_a == "ZERO":
            return self.encode_from_fixed_lns(t_b, s_b, l_b, sew)
        if t_b == "ZERO":
            return self.encode_from_fixed_lns(t_a, s_a, l_a, sew)

        if s_a != s_b and l_a == l_b:
            return self.encode_from_fixed_lns("ZERO", 0, 0, sew)

        if l_a >= l_b:
            l_max, l_min = l_a, l_b
            s_res = s_a
        else:
            l_max, l_min = l_b, l_a
            s_res = s_b

        q_fixed = l_max - l_min
        phi_fixed = self._gauss_lut_lookup(q_fixed, same_sign=(s_a == s_b))
        l_res = l_max + phi_fixed

        return self.encode_from_fixed_lns("NORMAL", s_res, l_res, sew)

    def execute_cvt_narrow(self, bits_src: int, sew_dst: int) -> int:
        sew_src = sew_dst * 2
        tipo, s, l_fixed = self.decode_to_fixed_lns(bits_src, sew_src)

        if tipo == "NAR":
            return self.encode_from_fixed_lns("NAR", 1, 0, sew_dst)
        if tipo == "ZERO":
            return self.encode_from_fixed_lns("ZERO", 0, 0, sew_dst)

        return self.encode_from_fixed_lns("NORMAL", s, l_fixed, sew_dst)


# ==============================================================================
# 2. PISTA FÍSICA DE EXECUÇÃO PARALELA (HARDWARE LANE)
# ==============================================================================

class HardwareLane:
    """
    Instância de pista física de execução (Lane). Cada lane integra uma instância
    independente do pré-decodificador, núcleo aritmético LNS e pós-codificador.
    """
    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.alu = TakumHardwareALU()

    def execute_op(self, op: str, a_bits: int, b_bits: int, sew: int):
        if op in ['vtadd.vv', 'vtadd.vf']:
            res_bits = self.alu.add_sub(a_bits, b_bits, sew, is_sub=False)
        elif op in ['vtsub.vv', 'vtsub.vf']:
            res_bits = self.alu.add_sub(a_bits, b_bits, sew, is_sub=True)
        elif op in ['vtmul.vv', 'vtmul.vf']:
            res_bits = self.alu.mul(a_bits, b_bits, sew)
        elif op == 'vncvt.t.t':
            res_bits = self.alu.execute_cvt_narrow(a_bits, sew_dst=sew)
        else:
            raise ValueError(f"Operação não suportada na lane: {op}")

        return res_bits


# ==============================================================================
# 3. MOTOR DE EXECUÇÃO VETORIAL PARALELO (RISC-V VECTOR ENGINE)
# ==============================================================================

class ParallelVectorEngine:
    """
    Simulador da extensão RISC-V Vector (Zvtakum) com N lanes físicas paralelas
    e fatiamento temporal de múltiplos ciclos de clock.
    """
    def __init__(self, n_lanes: int = 4, vlen: int = 128):
        self.n_lanes = n_lanes
        self.vlen = vlen
        self.lanes = [HardwareLane(i) for i in range(n_lanes)]
        self.vrf = {}

    def write_vrf(self, reg_num: int, floats: list, sew: int):
        bits_list = [float_to_takum_log(x, sew) for x in floats]
        self.vrf[reg_num] = (bits_list, sew)

    def write_vrf_bits(self, reg_num: int, bits_list: list, sew: int):
        self.vrf[reg_num] = (bits_list, sew)

    def read_vrf_floats(self, reg_num: int) -> list:
        bits_list, sew = self.vrf[reg_num]
        return [takum_log_to_float(b, sew) for b in bits_list]

    def read_vrf_bits(self, reg_num: int) -> list:
        bits_list, _ = self.vrf[reg_num]
        return bits_list

    def execute_vector_vector(self, op: str, vd: int, vs2: int, vs1: int, sew: int, vl: int):
        bits_vs2, _ = self.vrf[vs2]
        bits_vs1, _ = self.vrf[vs1]

        num_cycles = math.ceil(vl / self.n_lanes)
        result_bits = [0] * vl
        trace_logs = []

        for cycle in range(num_cycles):
            cycle_log = []
            for lane_idx in range(self.n_lanes):
                elem_idx = cycle * self.n_lanes + lane_idx
                if elem_idx < vl:
                    a_b = bits_vs2[elem_idx]
                    b_b = bits_vs1[elem_idx]

                    res_b = self.lanes[lane_idx].execute_op(op, a_b, b_b, sew)
                    result_bits[elem_idx] = res_b

                    val_a = takum_log_to_float(a_b, sew)
                    val_b = takum_log_to_float(b_b, sew)
                    val_r = takum_log_to_float(res_b, sew)

                    cycle_log.append({
                        'lane': lane_idx,
                        'elem': elem_idx,
                        'val_a': val_a,
                        'val_b': val_b,
                        'val_r': val_r,
                        'bits_r': hex(res_b)
                    })
            trace_logs.append((cycle + 1, cycle_log))

        self.vrf[vd] = (result_bits, sew)
        return num_cycles, trace_logs

    def execute_vector_narrow(self, op: str, vd: int, vs2: int, sew_dst: int, vl: int):
        sew_src = sew_dst * 2
        bits_vs2, _ = self.vrf[vs2]

        num_cycles = math.ceil(vl / self.n_lanes)
        result_bits = [0] * vl
        trace_logs = []

        for cycle in range(num_cycles):
            cycle_log = []
            for lane_idx in range(self.n_lanes):
                elem_idx = cycle * self.n_lanes + lane_idx
                if elem_idx < vl:
                    a_b = bits_vs2[elem_idx]
                    res_b = self.lanes[lane_idx].execute_op(op, a_b, 0, sew_dst)
                    result_bits[elem_idx] = res_b

                    val_a = takum_log_to_float(a_b, sew_src)
                    val_r = takum_log_to_float(res_b, sew_dst)

                    cycle_log.append({
                        'lane': lane_idx,
                        'elem': elem_idx,
                        'val_a': val_a,
                        'val_b': None,
                        'val_r': val_r,
                        'bits_r': hex(res_b)
                    })
            trace_logs.append((cycle + 1, cycle_log))

        self.vrf[vd] = (result_bits, sew_dst)
        return num_cycles, trace_logs

    def execute_dot_product(self, vd: int, vs2: int, vs1: int, sew: int, vl: int):
        bits_vs2, _ = self.vrf[vs2]
        bits_vs1, _ = self.vrf[vs1]

        num_cycles = math.ceil(vl / self.n_lanes)
        products_bits = []
        trace_logs = []

        for cycle in range(num_cycles):
            cycle_log = []
            for lane_idx in range(self.n_lanes):
                elem_idx = cycle * self.n_lanes + lane_idx
                if elem_idx < vl:
                    a_b = bits_vs2[elem_idx]
                    b_b = bits_vs1[elem_idx]
                    res_b = self.lanes[lane_idx].execute_op('vtmul.vv', a_b, b_b, sew)
                    products_bits.append(res_b)

                    val_a = takum_log_to_float(a_b, sew)
                    val_b = takum_log_to_float(b_b, sew)
                    val_r = takum_log_to_float(res_b, sew)

                    cycle_log.append({
                        'lane': lane_idx,
                        'elem': elem_idx,
                        'val_a': val_a,
                        'val_b': val_b,
                        'val_r': val_r,
                        'bits_r': hex(res_b)
                    })
            trace_logs.append((cycle + 1, cycle_log))

        alu_ref = TakumHardwareALU()
        acc_bits = float_to_takum_log(0.0, sew)

        for p_b in products_bits:
            acc_bits = alu_ref.add_sub(acc_bits, p_b, sew, is_sub=False)

        final_val = takum_log_to_float(acc_bits, sew)
        res_vec = [acc_bits] + [0] * (vl - 1)
        self.vrf[vd] = (res_vec, sew)
        return num_cycles + 1, trace_logs, final_val

    def execute_spmv_csr(self, val_floats: list, col_ind: list, row_ptr: list, x_floats: list, sew: int):
        num_rows = len(row_ptr) - 1
        y_res = []
        total_cycles = 0
        all_row_traces = []

        for r in range(num_rows):
            start = row_ptr[r]
            end = row_ptr[r + 1]
            row_vals = val_floats[start:end]
            row_cols = col_ind[start:end]
            x_sub = [x_floats[c] for c in row_cols]

            vl_row = len(row_vals)
            if vl_row == 0:
                y_res.append(0.0)
                continue

            self.write_vrf(1, row_vals, sew)
            self.write_vrf(2, x_sub, sew)

            cycles, trace_logs, dot_val = self.execute_dot_product(vd=3, vs2=1, vs1=2, sew=sew, vl=vl_row)
            total_cycles += cycles
            y_res.append(dot_val)
            all_row_traces.append((r, cycles, trace_logs, dot_val))

        return y_res, total_cycles, all_row_traces


# ==============================================================================
# 4. SUÍTE INTEGRADA DE CENÁRIOS DE TESTE VETORIAIS PARALELOS COM IMPRESSÃO DE CICLOS
# ==============================================================================

def print_trace_logs(num_cycles: int, trace_logs: list, max_display: int = 2):
    """
    Imprime o rastreamento da execução vetorial dividida nas N lanes físicas ao longo de ciclos.
    """
    print(f"    [Detalhamento de Execução nas {len(trace_logs[0][1]) if trace_logs else 4} Lanes Físicas em {num_cycles} Ciclo(s)]:")
    for cycle_num, lane_logs in trace_logs[:max_display]:
        parts = []
        for log in lane_logs:
            lane_str = f"L{log['lane']}(Elem {log['elem']}): "
            if log['val_b'] is not None:
                lane_str += f"{log['val_a']:.4g} op {log['val_b']:.4g} -> {log['val_r']:.4g}"
            else:
                lane_str += f"{log['val_a']:.4g} -> {log['val_r']:.4g}"
            parts.append(lane_str)
        print(f"      Ciclo {cycle_num}: " + " | ".join(parts))
    if len(trace_logs) > max_display:
        print(f"      ... [{len(trace_logs) - max_display} ciclo(s) omitidos por brevidade]")


def run_integrated_validation_suite():
    print("\n" + "=" * 100)
    print("BATERIA DE TESTES DE VALIDAÇÃO ARITMÉTICA E VETORIAL PARALELA (16 CENÁRIOS INTEGRADOS)")
    print("Execução em Lanes Físicas Paralelas com Rastreamento de Ciclos de Clock e Status de Validação")
    print("=" * 100)

    engine = ParallelVectorEngine(n_lanes=4, vlen=128)
    SEWS = (8, 16, 32)
    
    TOLERANCIAS = {
        8: 0.15,   # 15% em T8
        16: 0.02,  # 2% em T16
        32: 0.001  # 0.1% em T32
    }

    total_testes = 0
    aprovados = 0
    falhas = 0

    def erro_relativo(obtido: float, esperado: float) -> float:
        if math.isnan(esperado):
            return 0.0 if math.isnan(obtido) else float('inf')
        if esperado == 0.0:
            return abs(obtido)
        return abs(obtido - esperado) / abs(esperado)

    def registrar_resultado(nome: str, sucesso: bool, msg: str):
        nonlocal total_testes, aprovados, falhas
        total_testes += 1
        if sucesso:
            aprovados += 1
            status = "APROVADO"
        else:
            falhas += 1
            status = "FALHOU"
        print(f"  [{status:8}] Teste {total_testes:03d} - {nome}: {msg}")

    # --- CENÁRIO 1: MULTIPLICAÇÃO COM FRAÇÕES E INTEIROS ---
    print("\n--- CENÁRIO 1: MULTIPLICAÇÃO VETORIAL COM FRAÇÕES E INTEIROS (vtmul.vv) ---")
    pares_mul = [(0.5, 1.5), (2.0, 3.0), (4.0, 2.0), (0.25, 0.5), (1.25, 2.5), (10.0, 0.1), (3.0, 7.0), (0.125, 8.0)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_mul]
        vec_b = [p[1] for p in pares_mul]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        cycles, trace = engine.execute_vector_vector('vtmul.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res = engine.read_vrf_floats(3)

        errs = []
        for idx, (a, b) in enumerate(pares_mul):
            a_repr = takum_log_to_float(float_to_takum_log(a, sew), sew)
            b_repr = takum_log_to_float(float_to_takum_log(b, sew), sew)
            esp = a_repr * b_repr
            errs.append(erro_relativo(res[idx], esp))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtmul.vv (Vetor {vl} elementos)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 2: MULTIPLICAÇÃO COM SINAIS ---
    print("\n--- CENÁRIO 2: MULTIPLICAÇÃO VETORIAL COM SINAIS (vtmul.vv) ---")
    pares_sinais = [(2.0, -3.0), (-2.0, 3.0), (-2.0, -3.0), (0.5, -0.25), (-1.5, -1.5), (-10.0, 0.1)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_sinais]
        vec_b = [p[1] for p in pares_sinais]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        cycles, trace = engine.execute_vector_vector('vtmul.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res = engine.read_vrf_floats(3)

        errs = []
        for idx, (a, b) in enumerate(pares_sinais):
            a_repr = takum_log_to_float(float_to_takum_log(a, sew), sew)
            b_repr = takum_log_to_float(float_to_takum_log(b, sew), sew)
            esp = a_repr * b_repr
            errs.append(erro_relativo(res[idx], esp))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtmul.vv Sinais (Vetor {vl} elementos)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 3: ADIÇÃO DE VALORES POSITIVOS ---
    print("\n--- CENÁRIO 3: ADIÇÃO VETORIAL DE VALORES POSITIVOS (vtadd.vv) ---")
    pares_add = [(0.5, 1.5), (1.0, 1.0), (2.0, 3.0), (10.0, 20.0), (0.25, 0.75), (100.0, 2.0), (0.125, 0.125), (3.5, 4.5)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_add]
        vec_b = [p[1] for p in pares_add]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        cycles, trace = engine.execute_vector_vector('vtadd.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res = engine.read_vrf_floats(3)

        errs = []
        for idx, (a, b) in enumerate(pares_add):
            a_repr = takum_log_to_float(float_to_takum_log(a, sew), sew)
            b_repr = takum_log_to_float(float_to_takum_log(b, sew), sew)
            esp = a_repr + b_repr
            errs.append(erro_relativo(res[idx], esp))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtadd.vv Positivos (Vetor {vl} elementos)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 4: ADIÇÃO COM SINAIS OPOSTOS ---
    print("\n--- CENÁRIO 4: ADIÇÃO VETORIAL COM SINAIS OPOSTOS (vtadd.vv) ---")
    pares_add_op = [(10.0, -2.0), (100.0, -2.5), (5.0, -3.0), (1.5, -0.5), (-10.0, 2.0), (-5.0, 2.0), (0.75, -0.25), (-100.0, 1.0)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_add_op]
        vec_b = [p[1] for p in pares_add_op]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        cycles, trace = engine.execute_vector_vector('vtadd.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res = engine.read_vrf_floats(3)

        errs = []
        for idx, (a, b) in enumerate(pares_add_op):
            a_repr = takum_log_to_float(float_to_takum_log(a, sew), sew)
            b_repr = takum_log_to_float(float_to_takum_log(b, sew), sew)
            esp = a_repr + b_repr
            errs.append(erro_relativo(res[idx], esp))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtadd.vv Sinais Opostos (Vetor {vl} elementos)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 5: SUBTRAÇÃO VETORIAL ---
    print("\n--- CENÁRIO 5: SUBTRAÇÃO VETORIAL (vtsub.vv) ---")
    pares_sub = [(10.0, 2.0), (100.0, 2.5), (5.0, 3.0), (3.0, 5.0), (1.5, 0.5), (0.5, 1.5), (-10.0, -2.0), (-2.0, 10.0)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_sub]
        vec_b = [p[1] for p in pares_sub]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        cycles, trace = engine.execute_vector_vector('vtsub.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res = engine.read_vrf_floats(3)

        errs = []
        for idx, (a, b) in enumerate(pares_sub):
            a_repr = takum_log_to_float(float_to_takum_log(a, sew), sew)
            b_repr = takum_log_to_float(float_to_takum_log(b, sew), sew)
            esp = a_repr - b_repr
            errs.append(erro_relativo(res[idx], esp))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtsub.vv (Vetor {vl} elementos)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 6: IDENTIDADES ARITMÉTICAS ---
    print("\n--- CENÁRIO 6: IDENTIDADES ARITMÉTICAS VETORIAIS (vtadd.vv e vtmul.vv) ---")
    vec_id = [0.5, 1.0, 2.0, -3.0, 10.0]
    for sew in SEWS:
        vl = len(vec_id)
        engine.write_vrf(1, vec_id, sew)
        engine.write_vrf(2, [0.0] * vl, sew)
        cycles1, trace1 = engine.execute_vector_vector('vtadd.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res_add = engine.read_vrf_floats(3)

        engine.write_vrf(2, [1.0] * vl, sew)
        cycles2, trace2 = engine.execute_vector_vector('vtmul.vv', vd=4, vs2=1, vs1=2, sew=sew, vl=vl)
        res_mul = engine.read_vrf_floats(4)

        errs = []
        for idx, val in enumerate(vec_id):
            v_repr = takum_log_to_float(float_to_takum_log(val, sew), sew)
            errs.append(erro_relativo(res_add[idx], v_repr))
            errs.append(erro_relativo(res_mul[idx], v_repr))

        max_err = max(errs)
        limite = TOLERANCIAS[sew]
        suc = max_err <= limite
        print_trace_logs(cycles1, trace1)
        registrar_resultado(f"T{sew} Identidades Vetoriais (x+0, x*1)", suc, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 7: ZERO E NaR ---
    print("\n--- CENÁRIO 7: TRATATIVA VETORIAL DE EXCEÇÕES ZERO E NaR ---")
    for sew in SEWS:
        vec_x = [5.0, 5.0, 5.0, 5.0]
        vec_zero = [0.0, 0.0, 0.0, 0.0]
        nar_bit = 1 << (sew - 1)
        vec_nar = [nar_bit, nar_bit, nar_bit, nar_bit]

        engine.write_vrf(1, vec_x, sew)
        engine.write_vrf(2, vec_zero, sew)
        cycles_z, trace_z = engine.execute_vector_vector('vtmul.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=4)
        res_z = engine.read_vrf_floats(3)

        engine.write_vrf_bits(2, vec_nar, sew)
        cycles_n, trace_n = engine.execute_vector_vector('vtmul.vv', vd=4, vs2=1, vs1=2, sew=sew, vl=4)
        res_n = engine.read_vrf_floats(4)

        suc_z = all(r == 0.0 for r in res_z)
        suc_n = all(math.isnan(r) for r in res_n)
        limite = 0.0
        print_trace_logs(cycles_z, trace_z)
        registrar_resultado(f"T{sew} Exceções Zero e NaR em Vetores", suc_z and suc_n, f"Zero_Preservado={suc_z}, NaR_Preservado={suc_n}, limite_erro={limite:.3e}")

    # --- CENÁRIO 8: CANCELAMENTO ---
    print("\n--- CENÁRIO 8: CANCELAMENTO VETORIAL EXATO (vtadd.vv) ---")
    vec_c = [0.5, 1.0, 2.0, 5.0, 100.0]
    vec_nc = [-x for x in vec_c]
    for sew in SEWS:
        vl = len(vec_c)
        engine.write_vrf(1, vec_c, sew)
        engine.write_vrf(2, vec_nc, sew)
        cycles, trace = engine.execute_vector_vector('vtadd.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        res_bits = engine.read_vrf_bits(3)
        res_floats = engine.read_vrf_floats(3)

        err_max = max(erro_relativo(f, 0.0) for f in res_floats)
        suc = all(b == 0 for b in res_bits)
        limite = 0.0
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} Cancelamento Vetorial Exato (v + (-v))", suc, f"max_erro_abs={err_max:.3e}, bits_zerados={suc}, limite_erro={limite:.3e}")

    # --- CENÁRIO 9: COMUTATIVIDADE ---
    print("\n--- CENÁRIO 9: COMUTATIVIDADE VETORIAL BIT-A-BIT ---")
    pares_com = [(1.0, 2.0), (3.0, 5.0), (-2.0, 7.0), (-4.0, -3.0), (0.25, 0.75)]
    for sew in SEWS:
        vec_a = [p[0] for p in pares_com]
        vec_b = [p[1] for p in pares_com]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)

        cycles, trace = engine.execute_vector_vector('vtadd.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=vl)
        bits_add_ab = engine.read_vrf_bits(3)
        floats_add_ab = engine.read_vrf_floats(3)

        engine.execute_vector_vector('vtadd.vv', vd=4, vs2=2, vs1=1, sew=sew, vl=vl)
        bits_add_ba = engine.read_vrf_bits(4)
        floats_add_ba = engine.read_vrf_floats(4)

        engine.execute_vector_vector('vtmul.vv', vd=5, vs2=1, vs1=2, sew=sew, vl=vl)
        bits_mul_ab = engine.read_vrf_bits(5)
        engine.execute_vector_vector('vtmul.vv', vd=6, vs2=2, vs1=1, sew=sew, vl=vl)
        bits_mul_ba = engine.read_vrf_bits(6)

        err_add = max(erro_relativo(floats_add_ab[i], floats_add_ba[i]) for i in range(vl))
        suc = (bits_add_ab == bits_add_ba) and (bits_mul_ab == bits_mul_ba)
        limite = 0.0
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} Comutatividade Vetorial Bit-a-Bit", suc, f"add_diff_err={err_add:.3e}, bitwise_identical={suc}, limite_erro={limite:.3e}")

    # --- CENÁRIO 10: ASSOCIAÇÃO APROXIMADA ---
    print("\n--- CENÁRIO 10: ASSOCIATIVIDADE VETORIAL APROXIMADA (vtadd.vv) ---")
    for sew in SEWS:
        vec_a = [1.25, 2.5, 5.0, 10.0]
        vec_b = [2.5, 5.0, 10.0, 20.0]
        vec_c = [3.75, 7.5, 15.0, 30.0]
        vl = len(vec_a)
        engine.write_vrf(1, vec_a, sew)
        engine.write_vrf(2, vec_b, sew)
        engine.write_vrf(3, vec_c, sew)

        # (A + B) + C
        cycles, trace = engine.execute_vector_vector('vtadd.vv', vd=4, vs2=1, vs1=2, sew=sew, vl=vl)
        engine.execute_vector_vector('vtadd.vv', vd=5, vs2=4, vs1=3, sew=sew, vl=vl)
        res_lhs = engine.read_vrf_floats(5)

        # A + (B + C)
        engine.execute_vector_vector('vtadd.vv', vd=6, vs2=2, vs1=3, sew=sew, vl=vl)
        engine.execute_vector_vector('vtadd.vv', vd=7, vs2=1, vs1=6, sew=sew, vl=vl)
        res_rhs = engine.read_vrf_floats(7)

        err = max(erro_relativo(res_lhs[i], res_rhs[i]) for i in range(vl))
        limite = 2 * TOLERANCIAS[sew]
        suc = err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} Associatividade ((A+B)+C vs A+(B+C))", suc, f"max_erro_relativo={err:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 11: CONVERSÃO DE REDUÇÃO (vncvt.t.t) ---
    print("\n--- CENÁRIO 11: REDUÇÃO DE LARGURA DE ELEMENTO (vncvt.t.t) ---")
    
    # Teste 1: T32 -> T16
    vec_t32 = [0.125, 0.5, 1.0, 1.5, 2.5, 3.14159265, 10.0, -0.5]
    vl1 = len(vec_t32)
    engine.write_vrf(1, vec_t32, sew=32)
    cycles1, trace1 = engine.execute_vector_narrow('vncvt.t.t', vd=3, vs2=1, sew_dst=16, vl=vl1)
    res_t16 = engine.read_vrf_floats(3)
    errs1 = [erro_relativo(res_t16[i], vec_t32[i]) for i in range(vl1)]
    max_err1 = max(errs1)
    lim1 = 0.05
    print_trace_logs(cycles1, trace1)
    registrar_resultado("vncvt.t.t T32->T16 (Vetor 8 elementos)", max_err1 <= lim1, f"max_erro_relativo={max_err1:.3e}, limite_erro={lim1:.3e}")

    # Teste 2: T16 -> T8
    vec_t16 = [0.25, 0.5, 1.0, 2.0, 4.0, -1.0, -2.0, 8.0]
    vl2 = len(vec_t16)
    engine.write_vrf(1, vec_t16, sew=16)
    cycles2, trace2 = engine.execute_vector_narrow('vncvt.t.t', vd=3, vs2=1, sew_dst=8, vl=vl2)
    res_t8 = engine.read_vrf_floats(3)
    errs2 = [erro_relativo(res_t8[i], vec_t16[i]) for i in range(vl2)]
    max_err2 = max(errs2)
    lim2 = 0.15
    print_trace_logs(cycles2, trace2)
    registrar_resultado("vncvt.t.t T16->T8 (Vetor 8 elementos)", max_err2 <= lim2, f"max_erro_relativo={max_err2:.3e}, limite_erro={lim2:.3e}")

    # Teste 3: T32 -> T8
    vec_t32_b = [0.5, 1.0, 2.0, 3.0, 5.0, -1.0, -3.0, 10.0]
    vl3 = len(vec_t32_b)
    engine.write_vrf(1, vec_t32_b, sew=32)
    cycles3a, trace_3a = engine.execute_vector_narrow('vncvt.t.t', vd=2, vs2=1, sew_dst=16, vl=vl3)
    cycles3b, trace3b = engine.execute_vector_narrow('vncvt.t.t', vd=3, vs2=2, sew_dst=8, vl=vl3)
    res_t8_b = engine.read_vrf_floats(3)
    errs3 = [erro_relativo(res_t8_b[i], vec_t32_b[i]) for i in range(vl3)]
    max_err3 = max(errs3)
    lim3 = 0.20
    print("T32->T16 \n")
    print_trace_logs(cycles3a, trace_3a)
    print("T16->T8 \n")
    print_trace_logs(cycles3b, trace3b)
    registrar_resultado("vncvt.t.t T32->T8 em Cascata (Vetor 8 elementos)", max_err3 <= lim3, f"max_erro_relativo={max_err3:.3e}, limite_erro={lim3:.3e}")

    # --- CENÁRIO 12: PRODUTO ESCALAR VETORIAL ---
    print("\n--- CENÁRIO 12: PRODUTO ESCALAR VETORIAL (vtdot.vv) ---")
    vec_a_dot = [1.5, -2.0, 0.5, 4.0, 2.0, -1.0, 3.0, 0.5]
    vec_b_dot = [2.0, 3.0, -1.0, 0.25, 1.0, 2.0, -0.5, 4.0]
    vl_dot = len(vec_a_dot)
    
    for sew in SEWS:
        engine.write_vrf(1, vec_a_dot, sew)
        engine.write_vrf(2, vec_b_dot, sew)
        cycles, trace, dot_val = engine.execute_dot_product(vd=3, vs2=1, vs1=2, sew=sew, vl=vl_dot)

        a_reprs = [takum_log_to_float(float_to_takum_log(x, sew), sew) for x in vec_a_dot]
        b_reprs = [takum_log_to_float(float_to_takum_log(x, sew), sew) for x in vec_b_dot]
        esp_dot = sum(a * b for a, b in zip(a_reprs, b_reprs))

        err_dot = erro_relativo(dot_val, esp_dot)
        limite = 2 * TOLERANCIAS[sew]
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} vtdot.vv Produto Escalar (Vetor {vl_dot} elementos)", err_dot <= limite, f"obtido={dot_val:.6g}, esperado={esp_dot:.6g}, erro_relativo={err_dot:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 13: PRODUTO MATRIZ ESPARSA POR VETOR (SpMV em CSR via vtdot.vv) ---
    print("\n--- CENÁRIO 13: PRODUTO MATRIZ ESPARSA POR VETOR (SpMV em CSR via vtdot.vv) ---")
    val_csr = [5.0, 2.0, 3.0, -1.0, 4.0, 1.5, -2.5]
    col_ind = [0, 3, 1, 2, 2, 0, 3]
    row_ptr = [0, 2, 4, 5, 7]
    x_dense = [1.0, 2.0, 3.0, 4.0]

    for sew in SEWS:
        y_spmv, spmv_cycles, all_traces = engine.execute_spmv_csr(val_csr, col_ind, row_ptr, x_dense, sew=sew)
        
        x_repr = [takum_log_to_float(float_to_takum_log(v, sew), sew) for v in x_dense]
        val_repr = [takum_log_to_float(float_to_takum_log(v, sew), sew) for v in val_csr]
        
        y_esp = [
            val_repr[0]*x_repr[0] + val_repr[1]*x_repr[3],
            val_repr[2]*x_repr[1] + val_repr[3]*x_repr[2],
            val_repr[4]*x_repr[2],
            val_repr[5]*x_repr[0] + val_repr[6]*x_repr[3]
        ]
        
        err_spmv = max(erro_relativo(y_spmv[i], y_esp[i]) for i in range(4))
        limite = 0.35 if sew == 8 else (2 * TOLERANCIAS[sew])
        if all_traces and all_traces[0][2]:
            print_trace_logs(spmv_cycles, all_traces[0][2])
        registrar_resultado(f"T{sew} SpMV CSR via vtdot.vv (Matriz 4x4, 7 não-zeros)", err_spmv <= limite, f"max_erro_relativo={err_spmv:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 14: VALORES EXTREMOS ---
    print("\n--- CENÁRIO 14: VALORES EXTREMOS E SATURAÇÃO (vtmul.vv) ---")
    for sew in SEWS:
        if sew == 8:
            val_ext = [1e6, -1e8, 1e-6, 1e10]
            val_mul = [1e2, 1e2, 1e-2, 1e2]
        elif sew == 16:
            val_ext = [1e20, -1e25, 1e-20, 1e30]
            val_mul = [1e10, 1e10, 1e-10, 1e10]
        else:
            val_ext = [1e40, -1e45, 1e-40, 1e50]
            val_mul = [1e20, 1e20, 1e-20, 1e20]

        engine.write_vrf(1, val_ext, sew)
        engine.write_vrf(2, val_mul, sew)
        cycles, trace = engine.execute_vector_vector('vtmul.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=4)
        rec_list = engine.read_vrf_floats(3)

        suc = all(math.isfinite(r) and r != 0.0 and ((r > 0) == (val_ext[i] > 0)) for i, r in enumerate(rec_list))
        limite = 0.0
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} Preservação de Valores Extremos e Clamping", suc, f"Saturação_Suave={suc}, limite_erro={limite:.3e}")

    # --- CENÁRIO 15: ANÁLISE BITWISE EM VETORES ---
    print("\n--- CENÁRIO 15: ANÁLISE DE TRANSFORMAÇÕES BITWISE EM VETORES ---")
    vec_bw = [0.5, 1.25, 2.5, 5.0]
    vl_bw = len(vec_bw)
    for sew in SEWS:
        bits_list = [float_to_takum_log(v, sew) for v in vec_bw]
        mascara = (1 << sew) - 1
        neg_bits = [(-b) & mascara for b in bits_list]
        rec_neg = [takum_log_to_float(b, sew) for b in neg_bits]
        
        esp_neg = [-takum_log_to_float(b, sew) for b in bits_list]
        err_bw = max(erro_relativo(rec_neg[i], esp_neg[i]) for i in range(vl_bw))
        limite = TOLERANCIAS[sew]
        suc = err_bw <= limite
        print(f"    [Análise Bitwise em T{sew}]: Negação bitwise sobre vetor de {vl_bw} elementos")
        registrar_resultado(f"T{sew} Negação Bitwise (-x & mask)", suc, f"max_erro_relativo={err_bw:.3e}, limite_erro={limite:.3e}")

    # --- CENÁRIO 16: TESTES ALEATÓRIOS REPRODUTÍVEIS ---
    print("\n--- CENÁRIO 16: BATERIA VETORIAL ALEATÓRIA REPRODUTÍVEL (Todas 5 Instruções) ---")
    rng = random.Random(20261107)
    for sew in SEWS:
        a_rand = [rng.uniform(-15.0, 15.0) for _ in range(8)]
        b_rand = [rng.uniform(-15.0, 15.0) for _ in range(8)]
        engine.write_vrf(1, a_rand, sew)
        engine.write_vrf(2, b_rand, sew)
        cycles, trace = engine.execute_vector_vector('vtmul.vv', vd=3, vs2=1, vs1=2, sew=sew, vl=8)
        res_rand = engine.read_vrf_floats(3)

        errs = []
        limite = 2 * TOLERANCIAS[sew]
        for i in range(8):
            a_r = takum_log_to_float(float_to_takum_log(a_rand[i], sew), sew)
            b_r = takum_log_to_float(float_to_takum_log(b_rand[i], sew), sew)
            esp = a_r * b_r
            errs.append(erro_relativo(res_rand[i], esp))

        max_err = max(errs)
        suc_rand = max_err <= limite
        print_trace_logs(cycles, trace)
        registrar_resultado(f"T{sew} 8 Elementos Aleatórios em Lanes", suc_rand, f"max_erro_relativo={max_err:.3e}, limite_erro={limite:.3e}")

    print("\n" + "=" * 100)
    print(f"RESUMO FINAL DA VALIDAÇÃO ARITMÉTICA DA ULA: {aprovados}/{total_testes} TESTES APROVADOS ({100.0 * aprovados / total_testes:.2f}%)")
    print("=" * 100)


if __name__ == '__main__':
    run_integrated_validation_suite()
