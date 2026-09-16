package Puzzle8;
public class App {
    public static void main(String[] args) {
        
        String initialState = "7621 8435";
        String endState = "12345678 ";

        PuzzleTree searchTree = new PuzzleTree(initialState, endState);

        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long startTime = System.currentTimeMillis();

        System.out.println("=========================================");
        //Cola
        //searchTree.breadthFirstSearch(true);
        //Pila
        searchTree.breadthFirstSearch(false);

        long endTime = System.currentTimeMillis();
        long memoryUsed = runtime.totalMemory() - runtime.freeMemory();

        long timeElapsed = endTime - startTime;
        
        System.out.println("=========================================");
        System.out.println("End");
        System.out.println("Tiempo de ejecución: " + timeElapsed + " ms");
        
        double memoryInMB = memoryUsed / (1024.0 * 1024.0);
        System.out.printf("Memoria consumida: %.2f MB (%d bytes)%n", memoryInMB, memoryUsed);
        System.out.println("=========================================");
    }
    
}
