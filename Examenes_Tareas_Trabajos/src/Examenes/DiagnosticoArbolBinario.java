package Examenes;
// Clase que representa cada nodo del árbol
class Nodo {
    int valor;
    Nodo izquierdo;
    Nodo derecho;

    public Nodo(int valor) {
        this.valor = valor;
        this.izquierdo = null;
        this.derecho = null;
    }
}

public class DiagnosticoArbolBinario {
    Nodo raiz;

    public DiagnosticoArbolBinario() {
        this.raiz = null;
    }

    public void insertar(int valor) {
        raiz = insertarRecursivo(raiz, valor);
    }

    private Nodo insertarRecursivo(Nodo actual, int valor) {
        if (actual == null) {
            return new Nodo(valor);
        }

        if (valor < actual.valor) {
            actual.izquierdo = insertarRecursivo(actual.izquierdo, valor);
        } else if (valor > actual.valor) {
            actual.derecho = insertarRecursivo(actual.derecho, valor);
        }

        return actual;
    }

    public void inorden(Nodo nodo) {
        if (nodo != null) {
            inorden(nodo.izquierdo);
            System.out.print(nodo.valor + " ");
            inorden(nodo.derecho);
        }
    }

    public void preorden(Nodo nodo) {
        if (nodo != null) {
            System.out.print(nodo.valor + " ");
            preorden(nodo.izquierdo);
            preorden(nodo.derecho);
        }
    }

    public void postorden(Nodo nodo) {
        if (nodo != null) {
            postorden(nodo.izquierdo);
            postorden(nodo.derecho);
            System.out.print(nodo.valor + " ");
        }
    }

    public static void main(String[] args) {
        DiagnosticoArbolBinario arbol = new DiagnosticoArbolBinario();

        int[] elementos = {50, 30, 70, 20, 40, 60, 80};
        System.out.print("Insertando valores al árbol: ");
        for (int el : elementos) {
            System.out.print(el + " ");
            arbol.insertar(el);
        }
        System.out.println("\n");

        System.out.println("       RECORRIDOS DEL ÁRBOL     ");
        System.out.print("Inorden (Ordenado): ");
        arbol.inorden(arbol.raiz);
        System.out.println();

        System.out.print("Preorden          : ");
        arbol.preorden(arbol.raiz);
        System.out.println();

        System.out.print("Postorden         : ");
        arbol.postorden(arbol.raiz);
        System.out.println();
    }
}