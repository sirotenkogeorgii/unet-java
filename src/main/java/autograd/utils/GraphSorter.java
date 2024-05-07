package autograd.utils;

import autograd.Value;
import java.util.*;

/**
 * This class provides functionality to perform a topological sort on a computational graph.
 * The graph is represented by nodes of type {@link Value}, where each node can depend on other nodes.
 * The topological sort order is determined based on these dependencies.
 */
public class GraphSorter {

    private Set<Value> varsSeen;
    private List<Value> topSort;

    /**
     * Constructs a new GraphSorter instance.
     * Initializes the data structures used for keeping track of visited nodes and the order of nodes.
     */
    public GraphSorter() {
        this.varsSeen = new HashSet<>();
        this.topSort = new ArrayList<>();
    }

    /**
     * Performs a topological sort on the graph starting from the specified node.
     * It uses a depth-first search approach to traverse the graph.
     *
     * @param var The starting node for the topological sort.
     * @return A list of {@link Value} objects representing nodes in topologically sorted order.
     */
    public List<Value> topSort(Value var) {
        topSortHelper(var);
        Collections.reverse(topSort);
        return topSort;
    }

    /**
     * Helper method for {@link #topSort(Value)}. It recursively visits nodes,
     * marking them as seen and adding them to the topSort list.
     *
     * @param vr The node to process in this recursion step.
     */
    private void topSortHelper(Value vr) {
        if (!varsSeen.contains(vr)) {
            varsSeen.add(vr);
            for (Value pVar : vr.get_parents()) {
                topSortHelper(pVar);
            }
            topSort.add(vr);
        }
    }
}
