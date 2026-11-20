
import java.util.List;

public class Node{
    private String state;
    //private Boolean visited;
    private Node parent;
    //private List<Node> childs;
    private int depth;



    public Node(String state, Node parent){
        this.state = state;
        this.parent = parent;
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    public Node getParent() {
        return parent;
    }

    public void setParent(Node parent) {
        this.parent = parent;
    }


}
