package Puzzle8;

import java.util.Scanner;

public class App {

    public static void main(String[] args) {

        // Estado inicial
        String initialState = "7621 8435";

        // Estado final 
        String endState = "12345678 ";

        Scanner entrada = new Scanner(System.in);

        System.out.println("=========================================");
        System.out.println("           SOLUCIONADOR 8-PUZZLE");
        System.out.println("=========================================");

        System.out.println("Estado inicial:");
        Utils.formatState(initialState);

        System.out.println("Estado final:");
        Utils.formatState(endState);

        System.out.println("=========================================");
        System.out.println("Seleccione el algoritmo:");
        System.out.println("1. Primero en anchura");
        System.out.println("2. Costo uniforme");
        System.out.println("3. Primero en profundidad");
        System.out.println("4. Profundidad limitada");
        System.out.println("5. Profundidad iterativa");
        System.out.println("=========================================");

        System.out.print("Opcion: ");
        int opcion = entrada.nextInt();

        if (opcion < 1 || opcion > 5) {

            System.out.println("Opcion no valida.");

            entrada.close();

            return;
        }

        PuzzleTree searchTree =
                new PuzzleTree(initialState, endState);

        String nombreAlgoritmo = "";

        Runtime runtime = Runtime.getRuntime();

        runtime.gc();

        long memoriaInicial =
                runtime.totalMemory()
                - runtime.freeMemory();

        long startTime =
                System.currentTimeMillis();

        System.out.println();
        System.out.println("=========================================");

        switch (opcion) {

            case 1:

                nombreAlgoritmo =
                        "Primero en anchura";

                System.out.println(
                        "       PRIMERO EN ANCHURA"
                );

                System.out.println(
                        "========================================="
                );

                searchTree.breadthFirstSearch();

                break;

            case 2:

                nombreAlgoritmo =
                        "Costo uniforme";

                System.out.println(
                        "          COSTO UNIFORME"
                );

                System.out.println(
                        "========================================="
                );

                searchTree.uniformCostSearch();

                break;

            case 3:

                nombreAlgoritmo =
                        "Primero en profundidad";

                System.out.println(
                        "      PRIMERO EN PROFUNDIDAD"
                );

                System.out.println(
                        "========================================="
                );

                searchTree.depthFirstSearch();

                break;

            case 4:

                nombreAlgoritmo =
                        "Profundidad limitada";

                System.out.println(
                        "       PROFUNDIDAD LIMITADA"
                );

                System.out.println(
                        "========================================="
                );

                System.out.print(
                        "Ingrese el limite de profundidad: "
                );

                int limite = entrada.nextInt();

                searchTree.depthLimitedSearch(limite);

                break;

            case 5:

                nombreAlgoritmo =
                        "Profundidad iterativa";

                System.out.println(
                        "      PROFUNDIDAD ITERATIVA"
                );

                System.out.println(
                        "========================================="
                );

                searchTree.iterativeDeepeningSearch();

                break;
        }

        long endTime =
                System.currentTimeMillis();

        long memoriaFinal =
                runtime.totalMemory()
                - runtime.freeMemory();

        long memoriaUsada =
                memoriaFinal - memoriaInicial;

        if (memoriaUsada < 0) {
            memoriaUsada = 0;
        }

        long tiempoMilisegundos =
                endTime - startTime;

        double tiempoSegundos =
                tiempoMilisegundos / 1000.0;

        double memoriaMB =
                memoriaUsada / (1024.0 * 1024.0);

        // RESULTADOS

        System.out.println();
        System.out.println("=========================================");
        System.out.println("              RESULTADOS");
        System.out.println("=========================================");

        // Mostrar algoritmo utilizado
        System.out.println(
                "Algoritmo utilizado: "
                + nombreAlgoritmo
        );

        System.out.println(
                "Cantidad de movimientos/pasos: "
                + searchTree.getProfundidadSolucion()
        );

        System.out.printf(
                "Tiempo de ejecucion: %.3f segundos%n",
                tiempoSegundos
        );

        System.out.printf(
                "Memoria utilizada: %.2f MB%n",
                memoriaMB
        );

        System.out.println(
                "========================================="
        );

        entrada.close();
    }
}