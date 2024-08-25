/*
* 1. DFS - Path in Directed Graph
* 2. DFS - First Depth First Search
* 3. DFS - Cycle in Directed Graph
* 4. DFS - Number of Islands
* 5. DFS - Clone Graph
* 6. BFS - Rotten Oranges
* 7. BFS - Another BFS
* 8. BFS - Check Bipartite Graph
* 9. BFS - Valid Path
* 10. BFS - Number of Islands 2
* 11. BFS - Distance of Nearest Cell
* 12. BFS - Black Shapes
* 13. BFS - Maximum Depth
* 14. BFS - Shortest Distance in a Maze
* 15. BFS - Knight on Chess Board
* 16. Topological Sort
* 17. Dijkstra
* 18. Commutable Islands
* 19. Possibility of Finishing
 * */

//1. DFS - Path in Directed Graph

public class Solution {
    public int solve(int A, int[][] B) {
        List<List<Integer>> adj = new ArrayList<>();
        for(int i = 0; i <= A; i++){
            adj.add(new ArrayList<>());
        }
        for(int[] edge : B){
            adj.get(edge[0]).add(edge[1]);
        }

        boolean[] visited = new boolean[A+1];

        return dfs(1, A, adj, visited) ? 1: 0;
    }
    private boolean dfs(int node, int target, List<List<Integer>> adj, boolean[] visited){
        if(node == target){
            return true;
        }
        visited[node] = true;

        for(int neighbor : adj.get(node)){
            if(!visited[neighbor]){
                if(dfs(neighbor, target, adj, visited)){
                    return true;
                }
            }
        }
        return false;
    }
}



//* 2. DFS - First Depth First Search


public class Solution {
    public int solve(int[] A, final int B, final int C) {
        int N = A.length;
        List<List<Integer>> adjList = new ArrayList<>();
        for(int i = 0 ; i <= N; i++){
            adjList.add(new ArrayList<>());
        }
        for(int i = 1; i < N; i++){
            adjList.get(A[i]).add(i + 1);
        }

        boolean[] visited = new boolean[N + 1];
        return dfs(C, B, adjList, visited) ? 1:0;
    }
    private boolean dfs(int current, int target, List<List<Integer>> adjList, boolean[] visited){
        if(current == target){return true;}
        visited[current] = true;
        for(int neighbor : adjList.get(current)){
            if(!visited[neighbor]){
                if(dfs(neighbor, target, adjList, visited)){return true;}
            }
        }
        return false;
    }
}



//* 3. DFS - Cycle in Directed Graph

public class Solution {
    public int solve(int A, int[][] B) {
        //1. Create the adjacency List
        ArrayList<ArrayList<Integer>> adj = new ArrayList<>(A + 1);
        for(int i = 0 ; i <= A; i++){adj.add(new ArrayList<>());}
        for(int[] edge : B){adj.get(edge[0]).add(edge[1]);}

        //2. Initialise Visited and Recursion Array
        boolean[] visited = new boolean[A + 1];
        boolean[] recStack = new boolean[A + 1];

        //3. Perform DFS from each node
        for(int i = 1; i <= A; i++){
            if(!visited[i]){
                if(dfs(i, adj, visited, recStack)){return 1;}
            }
        }
        return 0;
    }

    private boolean dfs(int node, ArrayList<ArrayList<Integer>> adj, boolean[] visited, boolean[] recStack){
        visited[node] = true;
        recStack[node] = true;
        for(int neighbor : adj.get(node)){
            if(!visited[neighbor]){
                if(dfs(neighbor, adj, visited, recStack)){return true;}
            }else if(recStack[neighbor]){return true;}
        }
        recStack[node] = false;
        return false;
    }
}



//* 4. DFS - Number of Islands

public class Solution {
    public int solve(int[][] A) {
        if(A == null || A.length == 0){return 0;}
        int n = A.length;
        int m = A[0].length;
        boolean[][] visited = new boolean[n][m];
        int islandCount = 0;

        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(A[i][j] == 1 && !visited[i][j]){
                    dfs(A, visited, i, j);
                    islandCount++;
                }
            }
        }
        return islandCount;
    }

    private void dfs(int[][] A, boolean[][] visited, int i, int j){
        int[] rowDir = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] colDir = {-1, 0, 1, -1, 1, -1, 0, 1};

        visited[i][j] = true;
        for(int k = 0; k < 8; k++){
            int newRow = i + rowDir[k];
            int newCol = j + colDir[k];
            if(isValid(A, visited, newRow, newCol)){
                dfs(A, visited, newRow, newCol);
            }
        }
    }
    private boolean isValid(int[][] A, boolean[][] visited, int row, int col){
        return row >= 0 && row < A.length && col >= 0 && col < A[0].length && A[row][col] == 1 && !visited[row][col];
    }
}




//* 5. DFS - Clone Graph

public class Solution {
    private Map<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<>();
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node){
        // Handling null point
        if(node == null){return null;}
        // Check if Node is already cloned
        if(map.containsKey(node)){return map.get(node);}

        // Clone the Current Node
        UndirectedGraphNode clone = new UndirectedGraphNode(node.label);
        map.put(node, clone);

        //Recursively Clone Neighbors
        for(UndirectedGraphNode neighbor : node.neighbors){
            clone.neighbors.add(cloneGraph(neighbor));
        }
        return clone;

    }
}

//6. BFS - Rotten Oranges

public class Solution {
    public int solve(int[][] A) {
        //1. Initial Checks and Setup
        if(A == null || A.length == 0 || A[0].length == 0){return -1;}

        //2. Initialise Variables and Queue
        int N = A.length;
        int M = A[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int freshCount = 0;

        //3. Populate the Queue and Count Fresh Oranges
        for(int i = 0; i < N; i++){
            for(int j = 0; j < M; j++){
                if(A[i][j] == 2){queue.offer(new int[]{i, j});}
                else if(A[i][j] == 1){freshCount++;}
            }
        }

        // 4. Handle No fresh Oranges
        if(freshCount == 0){return 0;}

        // 5. BFS to spread Rot
        int[][] directions = {{0,1}, {1, 0}, {0, -1}, {-1, 0}};
        int minutes = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                int[] current = queue.poll();
                int x = current[0];
                int y = current[1];

                for(int[] dir: directions){
                    int newX = x + dir[0];
                    int newY = y + dir[1];

                    if(newX >= 0 && newX < N && newY >= 0 && newY < M && A[newX][newY] == 1){
                        A[newX][newY] = 2;
                        queue.offer(new int[]{newX, newY});
                        freshCount--;
                    }
                }
            }
            minutes++;
        }

        //6. Check Final Result
        return freshCount == 0? minutes -1: -1;
    }
}



//* 7. BFS - Another BFS

public class Solution {
    public int solve(int A, int[][] B, int C, int D) {
        //1. Build the Graph
        List<int[]>[] graph = new ArrayList[A];
        for (int i = 0; i < A; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] edge : B) {
            int u = edge[0];
            int v = edge[1];
            int weight = edge[2];
            graph[u].add(new int[]{v, weight});
            graph[v].add(new int[]{u, weight});
        }

        //2. Initialise Distances and Dequeue
        int[] distance = new int[A];
        Arrays.fill(distance, -1);
        distance[C] = 0;
        Deque<Integer> deque = new ArrayDeque<>();
        deque.addFirst(C);

        // 3. Perform Modified BFS
        while (!deque.isEmpty()) {
            int node = deque.pollFirst();

            for (int[] neighbor : graph[node]) {
                int nextNode = neighbor[0];
                int edgeWeight = neighbor[1];
                int newDistance = distance[node] + edgeWeight;

                if (distance[nextNode] == -1 || newDistance < distance[nextNode]) {
                    distance[nextNode] = newDistance;
                    if (edgeWeight == 1) {
                        deque.addFirst(nextNode);
                    } else {
                        deque.addLast(nextNode);
                    }
                }
            }
        }

        //4. Return the result

        return distance[D];
    }
}



//* 8. BFS - Check Bipartite Graph

public class Solution {
    public int solve(int A, int[][] B) {
        // Create adjacency list for the graph
        List<Integer>[] graph = new ArrayList[A];
        for (int i = 0; i < A; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] edge : B) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }

        // Array to store color of each node
        int[] color = new int[A];
        Arrays.fill(color, -1); // -1 means uncolored

        // Check each component of the graph
        for (int i = 0; i < A; i++) {
            if (color[i] == -1) { // If node i is uncolored, perform BFS
                if (!bfsCheck(graph, i, color)) {
                    return 0;
                }
            }
        }

        return 1; // Graph is bipartite
    }

    private boolean bfsCheck(List<Integer>[] graph, int start, int[] color) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        color[start] = 0; // Start with color 0

        while (!queue.isEmpty()) {
            int node = queue.poll();

            for (int neighbor : graph[node]) {
                if (color[neighbor] == -1) { // If uncolored, assign opposite color
                    color[neighbor] = 1 - color[node];
                    queue.offer(neighbor);
                } else if (color[neighbor] == color[node]) { // If the neighbor has the same color
                    return false; // Graph is not bipartite
                }
            }
        }

        return true; // Graph is bipartite
    }
}





//* 9. BFS - Valid Path

public class Solution {
    public String solve(int A, int B, int C, int D, int[] E, int[] F) {

        // Create a grid to keep track of obstructed cells
        boolean[][] obstructed = new boolean[A + 1][B + 1];

        // Mark obstructed cells based on circles
        for (int i = 0; i < C; i++) {
            int centerX = E[i];
            int centerY = F[i];
            for (int x = 0; x <= A; x++) {
                for (int y = 0; y <= B; y++) {
                    if (Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2) <= Math.pow(D, 2)) {
                        obstructed[x][y] = true;
                    }
                }
            }
        }

        // If start or end points are obstructed, return "NO" immediately
        if (obstructed[0][0] || obstructed[A][B]) {
            return "NO";
        }

        // BFS initialization
        boolean[][] visited = new boolean[A + 1][B + 1];
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{0, 0});
        visited[0][0] = true;

        // Directions for 8 possible moves
        int[][] directions = {
                {1, 0}, {0, 1}, {-1, 0}, {0, -1}, // Vertical and horizontal
                {1, 1}, {1, -1}, {-1, 1}, {-1, -1} // Diagonals
        };

        // BFS loop
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int cx = current[0];
            int cy = current[1];

            // Check if we reached the destination
            if (cx == A && cy == B) {
                return "YES";
            }

            // Explore all 8 directions
            for (int[] direction : directions) {
                int nx = cx + direction[0];
                int ny = cy + direction[1];

                // Check boundaries and if the cell is obstructed or visited
                if (nx >= 0 && nx <= A && ny >= 0 && ny <= B && !obstructed[nx][ny] && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.add(new int[]{nx, ny});
                }
            }
        }

        return "NO";}
}



//* 10. BFS - Number of Islands 2

public class Solution {
    private int rows, cols;
    private boolean[][] visited;
    private final int[] rowDirections = {-1, 1, 0, 0};
    private final int[] colDirections = {0, 0, -1, 1};

    private void bfs(int[][] A, int startX, int startY) {
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{startX, startY});
        visited[startX][startY] = true;

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int x = current[0];
            int y = current[1];

            for (int i = 0; i < 4; i++) {
                int newX = x + rowDirections[i];
                int newY = y + colDirections[i];

                if (newX >= 0 && newX < rows && newY >= 0 && newY < cols && !visited[newX][newY] && A[newX][newY] == 1) {
                    queue.add(new int[]{newX, newY});
                    visited[newX][newY] = true;
                }
            }
        }
    }
    public int solve(int[][] A) {
        if (A == null || A.length == 0) return 0;

        rows = A.length;
        cols = A[0].length;
        visited = new boolean[rows][cols];
        int islandCount = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (A[i][j] == 1 && !visited[i][j]) {
                    // Found a new island
                    islandCount++;
                    // Start BFS to mark all connected cells
                    bfs(A, i, j);
                }
            }
        }
        return islandCount;
    }
}




//* 11. BFS - Distance of Nearest Cell

public class Solution {
    public int[][] solve(int[][] A) {
        int N = A.length;
        int M = A[0].length;
        int[][] B = new int[N][M];
        int INF = N + M;  // A large value greater than the maximum possible distance

        // Initialize result matrix with infinity
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                B[i][j] = INF;
            }
        }

        // Queue for BFS
        Queue<int[]> queue = new LinkedList<>();

        // Enqueue all cells with 1 and set their distance to 0
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (A[i][j] == 1) {
                    B[i][j] = 0;
                    queue.offer(new int[]{i, j});
                }
            }
        }

        // Directions for moving in 4 possible directions
        int[] directions = {-1, 0, 1, 0, 0, -1, 0, 1};

        // Perform BFS
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0];
            int y = cell[1];

            for (int d = 0; d < 4; d++) {
                int nx = x + directions[d * 2];
                int ny = y + directions[d * 2 + 1];

                // Check if the new cell is within bounds
                if (nx >= 0 && nx < N && ny >= 0 && ny < M) {
                    if (B[nx][ny] > B[x][y] + 1) {
                        B[nx][ny] = B[x][y] + 1;
                        queue.offer(new int[]{nx, ny});
                    }
                }
            }
        }

        return B;
    }
}



//* 12. BFS - Black Shapes



public class Solution {
    public int black(String[] A) {
        int n = A.length;
        int m = A[0].length();

        boolean[][] visited = new boolean[n][m];
        int blackShapesCount = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (A[i].charAt(j) == 'X' && !visited[i][j]) {
                    // We found an unvisited 'X', so we start a BFS
                    bfs(A, visited, i, j, n, m);
                    blackShapesCount++;
                }
            }
        }
        return blackShapesCount;
    }

    private void bfs(String[] A, boolean[][] visited, int i, int j, int n, int m) {
        // Directions for moving in the matrix (right, down, left, up)
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{i, j});
        visited[i][j] = true;

        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            int x = cell[0];
            int y = cell[1];

            // Explore all 4 possible directions
            for (int[] dir : directions) {
                int newX = x + dir[0];
                int newY = y + dir[1];

                if (newX >= 0 && newX < n && newY >= 0 && newY < m && !visited[newX][newY] && A[newX].charAt(newY) == 'X') {
                    visited[newX][newY] = true;
                    queue.offer(new int[]{newX, newY});
                }
            }
        }
    }
}



//* 13. BFS - Maximum Depth




public class Solution {
    public int[] solve(int A, int[] B, int[] C, int[] D, int[] E, int[] F) {
        // Step 1: Build the tree using adjacency list
        List<Integer>[] adj = new ArrayList[A + 1];
        for (int i = 1; i <= A; i++) {
            adj[i] = new ArrayList<>();
        }

        for (int i = 0; i < A - 1; i++) {
            adj[B[i]].add(C[i]);
            adj[C[i]].add(B[i]);
        }

        // Step 2: Perform BFS to determine levels
        Map<Integer, List<Integer>> levelMap = new HashMap<>();
        int[] level = new int[A + 1];
        boolean[] visited = new boolean[A + 1];
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(1);
        visited[1] = true;
        level[1] = 0;

        int maxDepth = 0;

        while (!queue.isEmpty()) {
            int node = queue.poll();
            int currLevel = level[node];
            levelMap.computeIfAbsent(currLevel, k -> new ArrayList<>()).add(D[node - 1]);

            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    level[neighbor] = currLevel + 1;
                    queue.offer(neighbor);
                    maxDepth = Math.max(maxDepth, currLevel + 1);
                }
            }
        }

        // Step 3: Sort the nodes at each level for binary search
        for (int lvl : levelMap.keySet()) {
            Collections.sort(levelMap.get(lvl));
        }

        int Q = E.length;
        int[] result = new int[Q];

        // Step 4: Process each query
        for (int i = 0; i < Q; i++) {
            int L = E[i];
            int X = F[i];

            int effectiveLevel = L % (maxDepth + 1);
            List<Integer> nodesAtLevel = levelMap.getOrDefault(effectiveLevel, new ArrayList<>());

            // Binary search for the smallest element >= X
            int idx = Collections.binarySearch(nodesAtLevel, X);
            if (idx < 0) {
                idx = -(idx + 1); // get the insertion point
            }

            if (idx < nodesAtLevel.size()) {
                result[i] = nodesAtLevel.get(idx);
            } else {
                result[i] = -1;
            }
        }

        return result;
    }
}





//* 14. BFS - Shortest Distance in a Maze

public class Solution {
    private static final int[][] DIRECTIONS = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

    public int solve(int[][] A, int[] B, int[] C) {
        int N = A.length;
        int M = A[0].length;
        int[][] distance = new int[N][M];

        // Initialize distances to a large value
        for (int i = 0; i < N; i++) {
            Arrays.fill(distance[i], Integer.MAX_VALUE);
        }

        // BFS queue
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[]{B[0], B[1]});
        distance[B[0]][B[1]] = 0;

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int x = current[0];
            int y = current[1];

            // Explore all four directions
            for (int[] dir : DIRECTIONS) {
                int newX = x;
                int newY = y;
                int steps = 0;

                // Roll the ball until it hits a wall or boundary
                while (isValid(newX + dir[0], newY + dir[1], N, M, A)) {
                    newX += dir[0];
                    newY += dir[1];
                    steps++;
                }

                // If we found a shorter path to the new position
                if (distance[x][y] + steps < distance[newX][newY]) {
                    distance[newX][newY] = distance[x][y] + steps;
                    queue.offer(new int[]{newX, newY});
                }
            }
        }

        // Return the shortest distance to the destination
        return distance[C[0]][C[1]] == Integer.MAX_VALUE ? -1 : distance[C[0]][C[1]];
    }

    // Helper method to check if a position is within bounds and not a wall
    private boolean isValid(int x, int y, int N, int M, int[][] A) {
        return x >= 0 && y >= 0 && x < N && y < M && A[x][y] == 0;
    }
}




//* 15. BFS - Knight on Chess Board



public class Solution {
    // Define the possible moves for a knight
    private static final int[][] KNIGHT_MOVES = {
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
    };

    public int knight(int A, int B, int C, int D, int E, int F) {
        // Edge cases
        if (C == E && D == F) return 0;
        if (C < 1 || C > A || D < 1 || D > B || E < 1 || E > A || F < 1 || F > B) return -1;

        // Initialize BFS
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[A + 1][B + 1]; // Board is 1-indexed
        queue.offer(new int[]{C, D, 0}); // {x, y, distance}
        visited[C][D] = true;

        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            int x = current[0];
            int y = current[1];
            int distance = current[2];

            // Explore all possible knight moves
            for (int[] move : KNIGHT_MOVES) {
                int newX = x + move[0];
                int newY = y + move[1];

                // Check if the new position is within bounds and not visited
                if (isValid(newX, newY, A, B) && !visited[newX][newY]) {
                    // Check if we've reached the destination
                    if (newX == E && newY == F) {
                        return distance + 1;
                    }

                    // Mark as visited and add to queue
                    visited[newX][newY] = true;
                    queue.offer(new int[]{newX, newY, distance + 1});
                }
            }
        }

        // Destination is not reachable
        return -1;
    }

    // Helper method to check if the position is within the bounds of the board
    private boolean isValid(int x, int y, int A, int B) {
        return x >= 1 && x <= A && y >= 1 && y <= B;
    }
}






//16. Topological Sort

public class Solution {
    public int[] solve(int A, int[][] B) {
        // Initialize adjacency list and in-degree array
        List<List<Integer>> adjList = new ArrayList<>();
        int[] inDegree = new int[A + 1];
        for (int i = 0; i <= A; i++) {
            adjList.add(new ArrayList<>());
        }

        // Populate the adjacency list and in-degree array
        for (int[] edge : B) {
            adjList.get(edge[0]).add(edge[1]);
            inDegree[edge[1]]++;
        }

        // Use a priority queue to get the lexicographically smallest order
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i = 1; i <= A; i++) {
            if (inDegree[i] == 0) {
                pq.offer(i);
            }
        }

        List<Integer> topologicalOrder = new ArrayList<>();
        while (!pq.isEmpty()) {
            int node = pq.poll();
            topologicalOrder.add(node);
            for (int neighbor : adjList.get(node)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    pq.offer(neighbor);
                }
            }
        }

        // If the topological sort contains all nodes, return it; otherwise, return an empty array
        if (topologicalOrder.size() == A) {
            return topologicalOrder.stream().mapToInt(i -> i).toArray();
        } else {
            return new int[0];
        }
    }
}




//* 17. Dijkstra

public class Solution {
    //1. Define the Node Class
    static class Node implements Comparable<Node> {
        int vertex;
        int distance;

        Node(int vertex, int distance) {
            this.vertex = vertex;
            this.distance = distance;
        }

        public int compareTo(Node other) {
            return Integer.compare(this.distance, other.distance);
        }
    }
    public static int[] solve(int A, int[][] B, int C){
        //2. Initialize the Adjacency List
        List<List<Node>> adjList = new ArrayList<>();
        for (int i = 0; i < A; i++) {
            adjList.add(new ArrayList<>());
        }

        for (int[] edge : B) {
            int u = edge[0];
            int v = edge[1];
            int weight = edge[2];
            adjList.get(u).add(new Node(v, weight));
            adjList.get(v).add(new Node(u, weight)); // because the graph is undirected
        }

        //3. Initialize Dijkstra's Algorithm

        int[] distances = new int[A];
        Arrays.fill(distances, Integer.MAX_VALUE);
        distances[C] = 0;

        PriorityQueue<Node> minHeap = new PriorityQueue<>();
        minHeap.add(new Node(C, 0));

        boolean[] visited = new boolean[A];

        //4. Perform Dijkstra's Algorithm

        while (!minHeap.isEmpty()) {
            Node current = minHeap.poll();
            int u = current.vertex;

            if (visited[u]) continue;

            visited[u] = true;

            for (Node neighbor : adjList.get(u)) {
                int v = neighbor.vertex;
                int weight = neighbor.distance;

                if (!visited[v] && distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    minHeap.add(new Node(v, distances[v]));
                }
            }
        }

        //5. Handle Unreachable Nodes
        for (int i = 0; i < A; i++) {
            if (distances[i] == Integer.MAX_VALUE) {
                distances[i] = -1;
            }
        }
        return distances;
        //.
    }
}



//* 18. Commutable Islands


public class Solution {
    //1. Step 1 - Define the Edge Case
    static class Edge {
        int vertex;
        int cost;

        Edge(int vertex, int cost) {
            this.vertex = vertex;
            this.cost = cost;
        }
    }
    public static int solve(int A, int[][] B){
        //2. Build the Adjacency List
        List<List<Edge>> adjList = new ArrayList<>();
        for (int i = 0; i <= A; i++) {
            adjList.add(new ArrayList<>());
        }

        for (int[] edge : B) {
            int u = edge[0];
            int v = edge[1];
            int cost = edge[2];
            adjList.get(u).add(new Edge(v, cost));
            adjList.get(v).add(new Edge(u, cost));
        }

        //3.Implement Prim's Algorithm
        PriorityQueue<Edge> minHeap = new PriorityQueue<>((e1, e2) -> e1.cost - e2.cost);
        boolean[] inMST = new boolean[A + 1];
        int mstCost = 0;
        int edgesUsed = 0;

        // Start from vertex 1 (or any other arbitrary vertex)
        minHeap.offer(new Edge(1, 0));

        while (!minHeap.isEmpty() && edgesUsed < A) {
            Edge current = minHeap.poll();

            int u = current.vertex;
            int cost = current.cost;

            // Skip this edge if it leads to an already included vertex
            if (inMST[u]) continue;

            // Include this vertex in the MST
            inMST[u] = true;
            mstCost += cost;
            edgesUsed++;

            // Add all edges from this vertex to the heap
            for (Edge edge : adjList.get(u)) {
                if (!inMST[edge.vertex]) {
                    minHeap.offer(edge);
                }
            }
        }

        return mstCost;
    }
}



//* 19. Possibility of Finishing


public class Solution {
    public int solve(int A, int[] B, int[] C) {
        // Step 1: Build the graph
        List<List<Integer>> adjList = new ArrayList<>();
        int[] inDegree = new int[A + 1];

        // Initialize adjacency list
        for (int i = 0; i <= A; i++) {
            adjList.add(new ArrayList<>());
        }

        // Build the graph and fill the in-degree array
        for (int i = 0; i < B.length; i++) {
            int u = B[i];
            int v = C[i];
            adjList.get(u).add(v);
            inDegree[v]++;
        }

        // Step 2: Topological Sort using BFS
        Queue<Integer> queue = new LinkedList<>();

        // Enqueue all nodes with in-degree 0
        for (int i = 1; i <= A; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }

        int count = 0;

        // BFS
        while (!queue.isEmpty()) {
            int node = queue.poll();
            count++;

            for (int neighbor : adjList.get(node)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }

        // If count equals A, then all courses can be completed
        return count == A ? 1 : 0;
    }
}
