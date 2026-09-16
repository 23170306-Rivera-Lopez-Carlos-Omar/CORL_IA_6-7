package Puzzle8;

import java.util.HashSet;
import java.util.Set;
import java.util.List;
import java.util.Queue;
import java.util.LinkedList;
import java.util.Stack;

public class PuzzleTree {

    private String endState;
    private NodoPuzzle root;
    
    public PuzzleTree(String initialState, String endState){
        this.endState = endState;
        root = new NodoPuzzle(initialState, null);
    }

    public void breadthFirstSearch(boolean FIFO){
        Set<String> visited = new HashSet<>();
        NodoPuzzle current = root;
        visited.add(current.getState());

        if(FIFO){
            // BÚSQUEDA EN ANCHURA (BFS) - Cola de Java (Queue)
            Queue<NodoPuzzle> cola = new LinkedList<>();
            cola.add(current);
            
            while(!cola.isEmpty()){
                current = cola.poll(); // Saca el frente de la cola
                
                if(current.getState().compareTo(endState) == 0){
                    printPath(current);
                    System.out.println("Meta encontrada en la profundidad: " + current.getDepth());
                    break;  
                }
                
                // Generar hijos y recorrer la lista de Java
                List<NodoPuzzle> children = current.generateChildren();
                for (NodoPuzzle child : children) {
                    String childState = child.getState();
                    if (!visited.contains(childState)){
                        visited.add(childState);
                        cola.add(child);
                    }
                }
            }
        }
        else {
            // BÚSQUEDA EN PROFUNDIDAD (DFS) - Pila de Java (Stack)
            Stack<NodoPuzzle> pila = new Stack<>();
            pila.push(current);
            
            while(!pila.isEmpty()){
                current = pila.pop(); // Saca el tope de la pila
                
                if(current.getState().compareTo(endState) == 0){
                    printPath(current);
                    System.out.println("Meta encontrada en la profundidad: " + current.getDepth());
                    break;  
                }
                
                // Generar hijos y recorrer la lista de Java
                List<NodoPuzzle> children = current.generateChildren();
                for (NodoPuzzle child : children) {
                    String childState = child.getState();
                    if (!visited.contains(childState)){
                        pila.push(child);
                        visited.add(childState);
                    }
                }
            }
        }
    }

    private void printPath(NodoPuzzle nodo) {
        // Usamos la Pila nativa de Java
        Stack<String> pathStack = new Stack<>();
        NodoPuzzle current = nodo;
        
        while (current != null) {
            pathStack.push(current.getState());
            current = current.getParent();
        }
        
        while(!pathStack.isEmpty()) {
            // Sacamos (pop) e imprimimos en el orden correcto
            Utils.formatState(pathStack.pop());
        }
    }
}