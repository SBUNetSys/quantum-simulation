import numpy as np
import netsquid.qubits.operators as op
import netsquid.qubits.qubitapi as qapi


def controlled_unitary(number_of_qubits):
    """Create a controlled version of the unitary U."""
    n = number_of_qubits
    N = int(2 ** (n))  # dimension
    ## Quantum fourier transform
    Fourier_matrix = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            Fourier_matrix[k, j] = np.round(1 / np.sqrt(N) * np.exp(1j * 2 * np.pi / N * j * k), 12)
    # print(np.round(Fourier_matrix, 12))
    Fourier_matrix_inverse = np.conjugate(np.transpose(Fourier_matrix))
    # print(Fourier_matrix@Fourier_matrix_inverse)
    S1 = np.zeros((N, N), dtype=complex)
    S1_inverse = np.zeros((N, N), dtype=complex)
    for j in range(N):
        S1[j, j] = np.round(np.exp(1j * 2 * np.pi / N * (-j * j)), 12)
        S1_inverse[j, j] = np.round(np.exp(1j * 2 * np.pi / N * (j * j)), 12)
    # print(np.round(S1,10))
    S2 = np.round(Fourier_matrix @ S1_inverse @ Fourier_matrix_inverse, 12)
    S2_inverse = np.round(Fourier_matrix @ S1 @ Fourier_matrix_inverse, 12)

    T1 = S1
    T1_inverse = S1_inverse
    T3 = S2
    T3_inverse = S2_inverse

    Z1 = np.zeros((N, N), dtype=complex)
    Z1_inverse = np.zeros((N, N), dtype=complex)
    for j in range(N):
        Z1[j, j] = np.round(np.exp(1j * 2 * np.pi / N * j), 12)
        Z1_inverse[j, j] = np.round(np.exp(-1j * 2 * np.pi / N * j), 12)

    X1 = np.round(Fourier_matrix @ Z1 @ Fourier_matrix_inverse, 12)
    X1_inverse = np.round(Fourier_matrix @ Z1_inverse @ Fourier_matrix_inverse, 12)

    T2 = Z1 @ S1
    T2_inverse = S1_inverse @ Z1_inverse
    T4 = X1_inverse @ S2
    T4_inverse = S2_inverse @ X1
    # Now define the set of expander matrices
    Expander_unitaries = [T1, T2, T3, T4, T1_inverse, T2_inverse, T3_inverse, T4_inverse]
    # create 0 matrix with size of 2^n+3
    CU = np.zeros((2 ** (n + 3), 2 ** (n + 3)), dtype=complex)
    # perform diagonal operation
    block_size = 2 ** n
    for i in range(8):
        start = i * block_size
        end = (i + 1) * block_size
        CU[start:end, start:end] = Expander_unitaries[i]
    return CU



def measure_operator():
    """
    Create the measurement operators for the verification qubits m
    :return:
    """
    M0 = np.zeros((8, 8), dtype=complex)
    M0[0, 0] = 1
    M1 = np.eye(8) - M0
    return op.Operator("M0", M0), op.Operator("M1", M1)


if __name__ == '__main__':

    entangle_paris = []
    for i in range(8):
        qubit1, qubit2 = qapi.create_qubits(2)
        qapi.operate(qubit1, op.H)
        qapi.operate([qubit1, qubit2], op.CNOT)
        entangle_paris.append([qubit1, qubit2])
    teleport_entangle = []
    for i in range(3):
        qubit1, qubit2 = qapi.create_qubits(2)
        qapi.operate(qubit1, op.H)
        qapi.operate([qubit1, qubit2], op.CNOT)
        teleport_entangle.append([qubit1, qubit2])

    # qapi.operate(entangle_paris[0][1], op.X)
    # qapi.operate(entangle_paris[1][1], op.X)
    # qapi.operate(entangle_paris[2][1], op.X)
    # qapi.operate(entangle_paris[3][1], op.X)
    # qapi.operate(entangle_paris[4][1], op.X)
    # qapi.operate(entangle_paris[5][1], op.X)
    # qapi.operate(entangle_paris[6][1], op.X)
    # qapi.operate(entangle_paris[7][1], op.X)
    unitary_qubits = []
    for i in range(3):
        qubit, = qapi.create_qubits(1)
        qapi.operate(qubit, op.H)
        unitary_qubits.append(qubit)

    CU = controlled_unitary(8)
    # Example: Controlled-Y gate
    # Y = np.array([[0, -1j], [1j, 0]])
    CU_Gate = op.Operator("CU", CU)
    alice_qubits = unitary_qubits
    for i in entangle_paris:
        alice_qubits.append(i[0])
    qapi.operate(alice_qubits, CU_Gate)

    # teleportation
    need_teleport_qubit = unitary_qubits[:3]
    measurement_info = []
    for qubit_a, qubit_b in zip(need_teleport_qubit, teleport_entangle):
        qapi.operate([qubit_a, qubit_b[0]], op.CNOT)
        qapi.operate(qubit_a, op.H)
        m1, _ = qapi.measure(qubit_a)
        m2, _ = qapi.measure(qubit_b[0])
        measurement_info.append([m1, m2])

    # bob's side
    bob_qubits = []
    for result, qubit_b in zip(measurement_info, teleport_entangle):
        if result[0] == 1:
            qapi.operate(qubit_b[1], op.Z)
        if result[1] == 1:
            qapi.operate(qubit_b[1], op.X)
        bob_qubits.append(qubit_b[1])
    # CCU = np.conjugate(CU)
    CCU_Gate = CU_Gate.conj
    # bob_qubits = alice_qubits[:3]
    for i in entangle_paris:
        bob_qubits.append(i[1])
    qapi.operate(bob_qubits, CCU_Gate)

    for i in range(3):
        qapi.operate(bob_qubits[i], op.H)

    list_qubit = bob_qubits[:3]

    # for i in range(3):
    #     print(qapi.measure(list_qubit[i]))
    M0, M1 = measure_operator()
    print(qapi.gmeasure(list_qubit, [M0, M1]))
    #
    #
    # # Apply the controlled-Y operation
    # qapi.operate([control, target], CY)
