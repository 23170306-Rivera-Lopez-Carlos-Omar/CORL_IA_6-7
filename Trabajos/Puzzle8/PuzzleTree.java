package Puzzle8;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Queue;
import java.util.LinkedList;
import java.util.Stack;
import java.util.PriorityQueue;
import java.util.Comparator;

public class PuzzleTree {

    private String endState;
    private NodoPuzzle root;

    // Estadisticas
    private long nodosGenerados;
    private long nodosVisitados;
    private int profundidadSolucion;

    public PuzzleTree(String initialState, String endState) {

        this.endState = endState;
        root = new NodoPuzzle(initialState, null);

        reiniciarEstadisticas();
    }

    // =====================================================
    // REINICIAR ESTADISTICAS
    // =====================================================

    private void reiniciarEstadisticas() {

        nodosGenerados = 1;
        nodosVisitados = 0;
        profundidadSolucion = -1;
    }

    // =====================================================
    // 1. PRIMERO EN ANCHURA
    // =====================================================

    public void breadthFirstSearch() {

        reiniciarEstadisticas();

        Set<String> visited = new HashSet<>();

        Queue<NodoPuzzle> cola =
                new LinkedList<>();

        cola.add(root);

        visited.add(root.getState());

        while (!cola.isEmpty()) {

            NodoPuzzle current = cola.poll();

            nodosVisitados++;

            if (current.getState().equals(endState)) {

                profundidadSolucion =
                        current.getDepth();

                printPath(current);

                System.out.println(
                        "Meta encontrada."
                );

                return;
            }

            List<NodoPuzzle> children =
                    current.generateChildren();

            for (NodoPuzzle child : children) {

                nodosGenerados++;

                String childState =
                        child.getState();

                if (!visited.contains(childState)) {

                    visited.add(childState);

                    cola.add(child);
                }
            }
        }

        System.out.println(
                "No se encontro una solucion."
        );
    }

    // =====================================================
    // 2. COSTO UNIFORME
    // =====================================================

    public void uniformCostSearch() {

        reiniciarEstadisticas();

        Set<String> visited =
                new HashSet<>();

        PriorityQueue<NodoPuzzle> cola =
                new PriorityQueue<>(
                        new Comparator<NodoPuzzle>() {

                            @Override
                            public int compare(
                                    NodoPuzzle n1,
                                    NodoPuzzle n2) {

                                return Integer.compare(
                                        n1.getCost(),
                                        n2.getCost()
                                );
                            }
                        }
                );

        cola.add(root);

        while (!cola.isEmpty()) {

            NodoPuzzle current =
                    cola.poll();

            if (visited.contains(
                    current.getState())) {

                continue;
            }

            visited.add(current.getState());

            nodosVisitados++;

            if (current.getState().equals(endState)) {

                profundidadSolucion =
                        current.getDepth();

                printPath(current);

                System.out.println(
                        "Meta encontrada."
                );

                return;
            }

            List<NodoPuzzle> children =
                    current.generateChildren();

            for (NodoPuzzle child : children) {

                nodosGenerados++;

                if (!visited.contains(
                        child.getState())) {

                    cola.add(child);
                }
            }
        }

        System.out.println(
                "No se encontro una solucion."
        );
    }

    // =====================================================
    // 3. PRIMERO EN PROFUNDIDAD
    // =====================================================

    public void depthFirstSearch() {

        reiniciarEstadisticas();

        Set<String> visited =
                new HashSet<>();

        Stack<NodoPuzzle> pila =
                new Stack<>();

        pila.push(root);

        visited.add(root.getState());

        while (!pila.isEmpty()) {

            NodoPuzzle current =
                    pila.pop();

            nodosVisitados++;

            if (current.getState().equals(endState)) {

                profundidadSolucion =
                        current.getDepth();

                printPath(current);

                System.out.println(
                        "Meta encontrada."
                );

                return;
            }

            List<NodoPuzzle> children =
                    current.generateChildren();

            /*
             * Se agregan en orden inverso para
             * conservar un recorrido similar
             * al orden natural de los hijos.
             */

            for (int i = children.size() - 1;
                 i >= 0;
                 i--) {

                NodoPuzzle child =
                        children.get(i);

                nodosGenerados++;

                if (!visited.contains(
                        child.getState())) {

                    visited.add(
                            child.getState()
                    );

                    pila.push(child);
                }
            }
        }

        System.out.println(
                "No se encontro una solucion."
        );
    }

    // =====================================================
    // 4. PROFUNDIDAD LIMITADA
    // =====================================================

    public void depthLimitedSearch(int limit) {

        reiniciarEstadisticas();

        if (limit < 0) {

            System.out.println(
                    "El limite debe ser mayor o igual a 0."
            );

            return;
        }

        Set<String> visited =
                new HashSet<>();

        NodoPuzzle resultado =
                depthLimitedRecursive(
                        root,
                        limit,
                        visited
                );

        if (resultado != null) {

            profundidadSolucion =
                    resultado.getDepth();

            printPath(resultado);

            System.out.println(
                    "Meta encontrada."
            );

        } else {

            System.out.println(
                    "No se encontro la meta "
                    + "dentro del limite de profundidad."
            );
        }
    }

    private NodoPuzzle depthLimitedRecursive(
            NodoPuzzle current,
            int limit,
            Set<String> visited) {

        nodosVisitados++;

        if (current.getState().equals(endState)) {

            return current;
        }

        if (current.getDepth() >= limit) {

            return null;
        }

        visited.add(current.getState());

        List<NodoPuzzle> children =
                current.generateChildren();

        for (NodoPuzzle child : children) {

            nodosGenerados++;

            if (!visited.contains(
                    child.getState())) {

                NodoPuzzle resultado =
                        depthLimitedRecursive(
                                child,
                                limit,
                                visited
                        );

                if (resultado != null) {

                    return resultado;
                }
            }
        }

        visited.remove(current.getState());

        return null;
    }

    // =====================================================
    // 5. PROFUNDIDAD ITERATIVA
    // =====================================================

    public void iterativeDeepeningSearch() {

        reiniciarEstadisticas();

        int limit = 0;

        while (true) {

            Set<String> visited =
                    new HashSet<>();

            NodoPuzzle resultado =
                    depthLimitedRecursive(
                            root,
                            limit,
                            visited
                    );

            if (resultado != null) {

                profundidadSolucion =
                        resultado.getDepth();

                printPath(resultado);

                System.out.println(
                        "Meta encontrada."
                );

                System.out.println(
                        "Limite utilizado: "
                        + limit
                );

                return;
            }

            limit++;

            /*
             * El 8-Puzzle tiene como maximo
             * 31 movimientos en una solucion
             * optima desde un estado alcanzable.
             *
             * Este limite evita que el programa
             * siga indefinidamente en caso de
             * un problema.
             */

            if (limit > 40) {

                System.out.println(
                        "No se encontro la solucion."
                );

                return;
            }
        }
    }

    // =====================================================
    // IMPRIMIR CAMINO
    // =====================================================

    private void printPath(NodoPuzzle nodo) {

        Stack<String> pathStack =
                new Stack<>();

        NodoPuzzle current = nodo;

        while (current != null) {

            pathStack.push(
                    current.getState()
            );

            current =
                    current.getParent();
        }

        System.out.println();
        System.out.println(
                "CAMINO DE SOLUCION:"
        );

        System.out.println(
                "Cantidad de movimientos: "
                + nodo.getDepth()
        );

        System.out.println();

        int paso = 0;

        while (!pathStack.isEmpty()) {

            String state =
                    pathStack.pop();

            System.out.println(
                    "Paso " + paso + ":"
            );

            Utils.formatState(state);

            paso++;
        }
    }

    // =====================================================
    // GETTERS DE ESTADISTICAS
    // =====================================================

    public long getNodosGenerados() {

        return nodosGenerados;
    }

    public long getNodosVisitados() {

        return nodosVisitados;
    }

    public int getProfundidadSolucion() {

        return profundidadSolucion;
    }
}