
import java.util.LinkedList;
import java.util.Queue;

public class SearchTree {

    Node root = null;
    String initialState;
    String goalState;

    public SearchTree(String initialState, String goalState) {
        this.initialState = initialState;
        this.goalState = goalState;
        this.root = new Node(initialState, null);

    }

    public void breadFirstSearch() {
//1.bUSCAR EL NOOD RAIZ Y AGREGARLO A LA COLA
        Node currentNode;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

//2.Mientras la cola no este vacia, hacer lo siguiente

        while (!queue.isEmpty()) {
            //3.Sacar el primer nodo de la cola y verificar si es el objetivo
            currentNode = queue.poll();
            if (currentNode.getState().equals(goalState)) {
                System.out.println("Goal state found " + currentNode);
                return;
            }

            //4.Si no es el nood objetivo, generar sus hijos y agregar a la cola

        }


    }


}
