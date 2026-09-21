package Puzzle8;

import java.util.Scanner;

public class App {

    public static void main(String[] args) {

        String initialState = "7621 8435";
        String endState = "12345678 ";

        Scanner entrada = new Scanner(System.in);

        System.out.println("=========================================");
        System.out.println("          SOLUCIONADOR 8-PUZZLE");
        System.out.println("=========================================");
        System.out.println("1. Primero en anchura");
        System.out.println("2. Costo uniforme");
        System.out.println("3. Primero en profundidad");
        System.out.println("4. Profundidad limitada");
        System.out.println("5. Profundidad iterativa");
        System.out.println("=========================================");
        System.out.print("Seleccione un algoritmo: ");

        int opcion = entrada.nextInt();

        PuzzleTree searchTree = new PuzzleTree(initialState, endState);

        Runtime runtime = Runtime.getRuntime();

        runtime.gc();

        long memoriaInicial = runtime.totalMemory()
                - runtime.freeMemory();

        long startTime = System.currentTimeMillis();

        switch (opcion) {

            case 1:
                System.out.println("\n--- PRIMERO EN ANCHURA ---");
                searchTree.breadthFirstSearch();
                break;

            case 2:
                System.out.println("\n--- COSTO UNIFORME ---");
                searchTree.uniformCostSearch();
                break;

            case 3:
                System.out.println("\n--- PRIMERO EN PROFUNDIDAD ---");
                searchTree.depthFirstSearch();
                break;

            case 4:
                System.out.println("\n--- PROFUNDIDAD LIMITADA ---");
                searchTree.depthLimitedSearch(20);
                break;

            case 5:
                System.out.println("\n--- PROFUNDIDAD ITERATIVA ---");
                searchTree.iterativeDeepeningSearch();
                break;

            default:
                System.out.println("Opcion no valida.");
                return;
        }

        long endTime = System.currentTimeMillis();

        long memoriaFinal = runtime.totalMemory()
                - runtime.freeMemory();

        long memoriaUsada = memoriaFinal - memoriaInicial;

        System.out.println("\n=========================================");
        System.out.println("RESULTADOS");
        System.out.println("=========================================");
        System.out.println("Tiempo de ejecucion: "
                + (endTime - startTime) + " ms");

        double memoriaMB = memoriaUsada / (1024.0 * 1024.0);

        System.out.printf("Memoria utilizada: %.2f MB%n", memoriaMB);
        System.out.println("=========================================");
    }
}