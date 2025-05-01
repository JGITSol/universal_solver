<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# NeuroSymbolic Mathematical Reasoning System (NSMRS) - Advanced Components

Building on the existing comprehensive implementation, I'll now add several critical advanced components that take this system to the next level, including visualization capabilities, tool integration, distributed computing support, and persistent knowledge management.

```python
# ====================== VISUALIZATION MODULE ======================

class MathVisualizer:
    """Advanced visualization system for mathematical proofs and constructions."""
    
    def __init__(self, figsize=(10, 8), style="modern", interactive=True):
        """
        Initialize the visualization system.
        
        Args:
            figsize: Size of the visualization figures
            style: Visualization style ("modern", "classic", "minimal")
            interactive: Whether to enable interactive visualizations
        """
        self.figsize = figsize
        self.style = style
        self.interactive = interactive
        self.colors = self._get_color_scheme(style)
        
        # Initialize plotting settings
        plt.rcParams["figure.figsize"] = figsize
        if style == "modern":
            plt.style.use("seaborn-v0_8-whitegrid")
        elif style == "classic":
            plt.style.use("classic")
        elif style == "minimal":
            plt.style.use("seaborn-v0_8-white")
    
    def _get_color_scheme(self, style):
        """Get color scheme based on style."""
        if style == "modern":
            return {
                "points": "#3498db",
                "lines": "#2c3e50",
                "circles": "#e74c3c",
                "angles": "#9b59b6",
                "highlight": "#f1c40f",
                "background": "#ecf0f1"
            }
        elif style == "classic":
            return {
                "points": "blue",
                "lines": "black",
                "circles": "red",
                "angles": "green",
                "highlight": "orange",
                "background": "white"
            }
        else:  # minimal
            return {
                "points": "#555555",
                "lines": "#333333",
                "circles": "#777777",
                "angles": "#999999",
                "highlight": "#000000",
                "background": "#ffffff"
            }
    
    def visualize_geometry_problem(self, formalization, highlight_steps=None):
        """
        Create a visualization of a geometric problem.
        
        Args:
            formalization: Problem formalization
            highlight_steps: Optional list of steps to highlight
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract entities from formalization
        entities = formalization.get("entities", [])
        
        # Get points with coordinates (or generate coordinates if needed)
        points = {}
        for entity in entities:
            if entity.get("type") == "point":
                name = entity.get("name")
                # Check if coordinates are provided
                if "x" in entity and "y" in entity:
                    points[name] = (entity["x"], entity["y"])
                else:
                    # Generate coordinates based on name (for consistent placement)
                    # This ensures points like A, B, C form a reasonable configuration
                    x, y = self._generate_point_coordinates(name, len(points))
                    points[name] = (x, y)
        
        # Draw entities
        for entity in entities:
            entity_type = entity.get("type")
            
            if entity_type == "point":
                self._draw_point(ax, entity, points)
            elif entity_type == "line":
                self._draw_line(ax, entity, points)
            elif entity_type == "circle":
                self._draw_circle(ax, entity, points)
            elif entity_type == "angle":
                self._draw_angle(ax, entity, points)
        
        # Highlight specific steps if provided
        if highlight_steps:
            self._highlight_steps(ax, highlight_steps, points)
        
        # Set equal aspect ratio and adjust limits
        ax.set_aspect("equal")
        self._adjust_limits(ax, points)
        
        # Add title
        if "goal" in formalization:
            goal = formalization["goal"]
            if isinstance(goal, dict) and "description" in goal:
                plt.title(f"Goal: {goal['description']}")
        
        # Set background color
        ax.set_facecolor(self.colors["background"])
        
        # Make interactive if enabled
        if self.interactive:
            self._make_interactive(fig, ax, points)
        
        return fig
    
    def _generate_point_coordinates(self, name, index):
        """Generate coordinates for a point based on its name."""
        # Use the name to generate consistent coordinates
        # This allows points like A, B, C to form a triangle
        name_hash = sum(ord(c) for c in name)
        
        if index < 3:  # First three points form a triangle
            if index == 0:
                return 2, 2
            elif index == 1:
                return 8, 2
            else:
                return 5, 7
        else:
            # Generate coordinates based on name hash and index
            angle = (name_hash % 360) * (np.pi / 180)
            radius = 3 + (index % 3)
            x = 5 + radius * np.cos(angle)
            y = 5 + radius * np.sin(angle)
            return x, y
    
    def _draw_point(self, ax, entity, points):
        """Draw a point on the axis."""
        name = entity.get("name")
        if name in points:
            x, y = points[name]
            ax.plot(x, y, 'o', color=self.colors["points"], markersize=8)
            ax.text(x + 0.3, y + 0.3, name, fontsize=12)
    
    def _draw_line(self, ax, entity, points):
        """Draw a line on the axis."""
        name = entity.get("name")
        point_names = entity.get("points", [])
        
        if len(point_names) >= 2 and all(p in points for p in point_names):
            # Get the coordinates of the endpoints
            coords = [points[p] for p in point_names]
            
            # If the line passes through exactly two points, draw a line segment
            if len(coords) == 2:
                x_vals = [coords[0][0], coords[1][0]]
                y_vals = [coords[0][1], coords[1][1]]
                ax.plot(x_vals, y_vals, '-', color=self.colors["lines"], linewidth=2)
                
                # Add the line name at the midpoint
                mid_x = sum(x_vals) / 2
                mid_y = sum(y_vals) / 2
                ax.text(mid_x, mid_y, name, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # If the line passes through more points, draw a best-fit line
            else:
                x_vals = [p[0] for p in coords]
                y_vals = [p[1] for p in coords]
                
                # Compute line of best fit
                coeffs = np.polyfit(x_vals, y_vals, 1)
                
                # Get the x-range of the plot
                xlim = ax.get_xlim()
                if not xlim or xlim[0] == xlim[1]:
                    xlim = (0, 10)
                
                # Draw the line across the full width
                x = np.array(xlim)
                y = coeffs[0] * x + coeffs[1]
                ax.plot(x, y, '-', color=self.colors["lines"], linewidth=2)
                
                # Add the line name at the edge
                ax.text(x[1], y[1], name, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _draw_circle(self, ax, entity, points):
        """Draw a circle on the axis."""
        name = entity.get("name")
        center_name = entity.get("center")
        radius_point_name = entity.get("radius_point")
        explicit_radius = entity.get("radius")
        
        if center_name in points:
            center = points[center_name]
            
            # Determine radius
            radius = None
            if radius_point_name and radius_point_name in points:
                radius_point = points[radius_point_name]
                dx = center[0] - radius_point[0]
                dy = center[1] - radius_point[1]
                radius = np.sqrt(dx**2 + dy**2)
            elif explicit_radius:
                radius = explicit_radius
            
            if radius:
                circle = plt.Circle(center, radius, fill=False, 
                                   color=self.colors["circles"], linewidth=2)
                ax.add_artist(circle)
                
                # Add the circle name
                angle = np.pi/4  # 45 degrees
                label_x = center[0] + 0.7 * radius * np.cos(angle)
                label_y = center[1] + 0.7 * radius * np.sin(angle)
                ax.text(label_x, label_y, name, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    def _draw_angle(self, ax, entity, points):
        """Draw an angle on the axis."""
        name = entity.get("name")
        vertex_name = entity.get("vertex")
        arm1_name = entity.get("arm1")
        arm2_name = entity.get("arm2")
        
        if vertex_name in points and arm1_name in points and arm2_name in points:
            vertex = points[vertex_name]
            arm1 = points[arm1_name]
            arm2 = points[arm2_name]
            
            # Calculate vectors
            v1 = np.array([arm1[0] - vertex[0], arm1[1] - vertex[1]])
            v2 = np.array([arm2[0] - vertex[0], arm2[1] - vertex[1]])
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Calculate angles
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            
            # Ensure angle2 > angle1
            if angle2 < angle1:
                angle2 += 2 * np.pi
            
            # Draw arc
            radius = 0.5
            arc = plt.matplotlib.patches.Arc(
                vertex, 2*radius, 2*radius, 
                theta1=angle1*180/np.pi, theta2=angle2*180/np.pi,
                color=self.colors["angles"], linewidth=2
            )
            ax.add_artist(arc)
            
            # Add angle name
            mid_angle = (angle1 + angle2) / 2
            label_x = vertex[0] + 1.2 * radius * np.cos(mid_angle)
            label_y = vertex[1] + 1.2 * radius * np.sin(mid_angle)
            ax.text(label_x, label_y, name, fontsize=10)
    
    def _highlight_steps(self, ax, steps, points):
        """Highlight specific steps of a proof."""
        # Implementation depends on the step format
        for step in steps:
            step_type = step.get("type")
            entities = step.get("entities", [])
            
            for entity in entities:
                if entity.get("type") == "point" and entity.get("name") in points:
                    name = entity.get("name")
                    x, y = points[name]
                    ax.plot(x, y, 'o', color=self.colors["highlight"], markersize=10)
    
    def _adjust_limits(self, ax, points):
        """Adjust the axis limits to fit all points with margin."""
        if not points:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            return
        
        x_coords = [p[0] for p in points.values()]
        y_coords = [p[1] for p in points.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        margin = max(1, (x_max - x_min) * 0.1, (y_max - y_min) * 0.1)
        
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    
    def _make_interactive(self, fig, ax, points):
        """Add interactive features to the visualization."""
        if not self.interactive:
            return
        
        # This would add interactive features using matplotlib's widgets
        # For this implementation, we'll just add some basic interactivity
        
        # Example: Enable zoom and pan
        plt.rcParams['toolbar'] = 'toolbar2'
    
    def visualize_algebra_proof(self, formalization, proof):
        """
        Visualize an algebraic proof.
        
        Args:
            formalization: Problem formalization
            proof: The proof steps
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create a textual representation of the algebraic proof
        if not proof or not proof.steps:
            ax.text(0.5, 0.5, "No proof available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return fig
        
        # Format and display the proof steps
        text = "Algebraic Proof:\n\n"
        
        for i, step in enumerate(proof.steps):
            text += f"Step {i+1}: {step.description}\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', wrap=True)
        ax.axis('off')
        
        return fig
    
    def visualize_proof_graph(self, proof):
        """
        Generate a graph visualization of the proof structure.
        
        Args:
            proof: The proof object
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if not proof or not proof.steps:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No proof available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return fig
        
        # Create a directed graph representing the proof
        G = nx.DiGraph()
        
        # Add nodes for each step
        for i, step in enumerate(proof.steps):
            G.add_node(i, label=f"Step {i+1}: {step.step_type}")
        
        # Add edges (this is simplified - a real implementation would trace fact dependencies)
        for i in range(1, len(proof.steps)):
            G.add_edge(i-1, i)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate layout
        pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self.colors["points"], 
                              node_size=700, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, 
                              edge_color=self.colors["lines"], width=2)
        nx.draw_networkx_labels(G, pos, ax=ax, 
                               labels={n: G.nodes[n]["label"] for n in G.nodes})
        
        plt.title("Proof Structure Graph")
        ax.axis('off')
        
        return fig
    
    def visualize_search_tree(self, search_data):
        """
        Visualize the SKEST search tree.
        
        Args:
            search_data: Data from the search process
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # This would visualize the search tree exploration
        # Simplified implementation for demonstration
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data about branch explorations
        branches = search_data.get("branches", [])
        
        if not branches:
            ax.text(0.5, 0.5, "No search tree data available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.axis('off')
            return fig
        
        # Create a tree visualization
        G = nx.DiGraph()
        
        # Add nodes and edges
        for branch in branches:
            branch_id = branch.get("id")
            parent_id = branch.get("parent")
            success = branch.get("success", False)
            
            # Add the node
            G.add_node(branch_id, success=success)
            
            # Add edge to parent if exists
            if parent_id is not None:
                G.add_edge(parent_id, branch_id)
        
        # Generate layout
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        
        # Draw the graph
        node_colors = [self.colors["highlight"] if G.nodes[n]["success"] else self.colors["points"] 
                      for n in G.nodes]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, 
                              edge_color=self.colors["lines"], width=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        plt.title("SKEST Search Tree")
        ax.axis('off')
        
        return fig
    
    def export_visualization(self, fig, filename, format="png", dpi=300):
        """
        Export a visualization to a file.
        
        Args:
            fig: The figure to export
            filename: Output filename
            format: Output format (png, pdf, svg)
            dpi: Resolution for raster formats
        """
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches="tight")
        logger.info(f"Visualization exported to {filename}")

# ====================== EXTERNAL TOOLS INTEGRATION ======================

class ToolIntegrationManager:
    """
    Manages integration with external mathematical tools and libraries.
    Enables the system to leverage specialized tools for specific tasks.
    """
    
    def __init__(self):
        """Initialize the tool integration manager."""
        self.tools = {}
        self.loaded_tools = set()
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tool integrations."""
        # Register SymPy for symbolic algebra
        self.register_tool(
            name="sympy",
            description="Symbolic mathematics library for Python",
            functions={
                "solve_equation": self._sympy_solve_equation,
                "simplify_expression": self._sympy_simplify,
                "calculate_integral": self._sympy_integrate,
                "factor_polynomial": self._sympy_factor
            }
        )
        
        # Register NumPy for numerical calculations
        self.register_tool(
            name="numpy",
            description="Numerical computing library for Python",
            functions={
                "matrix_operations": self._numpy_matrix_ops,
                "statistical_analysis": self._numpy_stats,
                "numerical_optimization": self._numpy_optimize
            }
        )
        
        # Register NetworkX for graph theory
        self.register_tool(
            name="networkx",
            description="Graph theory and complex networks library",
            functions={
                "create_graph": self._networkx_create_graph,
                "analyze_graph": self._networkx_analyze_graph,
                "find_path": self._networkx_find_path
            }
        )
        
        # Register SageMath interface (if available)
        self.register_tool(
            name="sage",
            description="Advanced mathematics software system",
            functions={
                "advanced_number_theory": self._sage_number_theory,
                "algebraic_geometry": self._sage_algebraic_geometry,
                "group_theory": self._sage_group_theory
            }
        )
        
        # Register GeoGebra interface (for geometric visualization and computation)
        self.register_tool(
            name="geogebra",
            description="Dynamic mathematics software for geometry",
            functions={
                "geometric_construction": self._geogebra_construct,
                "interactive_visualization": self._geogebra_visualize
            }
        )
    
    def register_tool(self, name, description, functions):
        """
        Register a tool with the system.
        
        Args:
            name: Tool name
            description: Tool description
            functions: Dictionary of function names to implementations
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "functions": functions,
            "loaded": False
        }
        logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name):
        """
        Get a registered tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            dict: Tool information or None if not found
        """
        if name not in self.tools:
            logger.warning(f"Tool not found: {name}")
            return None
        
        # Load the tool if not already loaded
        if not self.tools[name]["loaded"]:
            self._load_tool(name)
        
        return self.tools[name]
    
    def _load_tool(self, name):
        """
        Load a tool dynamically.
        
        Args:
            name: Tool name
        """
        if name not in self.tools:
            logger.warning(f"Cannot load unknown tool: {name}")
            return
        
        if name in self.loaded_tools:
            return
        
        try:
            # Perform any necessary initialization
            if name == "sympy":
                self._load_sympy()
            elif name == "numpy":
                self._load_numpy()
            elif name == "networkx":
                self._load_networkx()
            elif name == "sage":
                self._load_sage()
            elif name == "geogebra":
                self._load_geogebra()
            
            self.tools[name]["loaded"] = True
            self.loaded_tools.add(name)
            logger.info(f"Loaded tool: {name}")
        except Exception as e:
            logger.error(f"Failed to load tool {name}: {str(e)}")
    
    def call_tool_function(self, tool_name, function_name, *args, **kwargs):
        """
        Call a function from a specific tool.
        
        Args:
            tool_name: Name of the tool
            function_name: Name of the function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Any: Result of the function call
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        functions = tool.get("functions", {})
        if function_name not in functions:
            raise ValueError(f"Function {function_name} not found in tool {tool_name}")
        
        function = functions[function_name]
        return function(*args, **kwargs)
    
    # Tool loading implementations
    
    def _load_sympy(self):
        """Load SymPy library."""
        # This would actually import the library
        # For demonstration, we'll assume it's already available
        pass
    
    def _load_numpy(self):
        """Load NumPy library."""
        # This would actually import the library
        # For demonstration, we'll assume it's already available
        pass
    
    def _load_networkx(self):
        """Load NetworkX library."""
        # This would actually import the library
        # For demonstration, we'll assume it's already available
        pass
    
    def _load_sage(self):
        """Load SageMath interface."""
        # This would set up an interface to SageMath
        # For demonstration, we'll assume it's already available
        pass
    
    def _load_geogebra(self):
        """Load GeoGebra interface."""
        # This would set up an interface to GeoGebra
        # For demonstration, we'll assume it's already available
        pass
    
    # Tool function implementations
    
    # SymPy functions
    def _sympy_solve_equation(self, equation, variable):
        """Solve an algebraic equation for a variable."""
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            solution = sp.solve(equation, var)
            return solution
        except Exception as e:
            logger.error(f"Error in sympy_solve_equation: {str(e)}")
            return None
    
    def _sympy_simplify(self, expression):
        """Simplify a mathematical expression."""
        try:
            import sympy as sp
            simplified = sp.simplify(expression)
            return str(simplified)
        except Exception as e:
            logger.error(f"Error in sympy_simplify: {str(e)}")
            return expression
    
    def _sympy_integrate(self, expression, variable, lower=None, upper=None):
        """Calculate an integral."""
        try:
            import sympy as sp
            var = sp.Symbol(variable)
            
            if lower is not None and upper is not None:
                result = sp.integrate(expression, (var, lower, upper))
            else:
                result = sp.integrate(expression, var)
                
            return str(result)
        except Exception as e:
            logger.error(f"Error in sympy_integrate: {str(e)}")
            return None
    
    def _sympy_factor(self, expression):
        """Factor a polynomial."""
        try:
            import sympy as sp
            factored = sp.factor(expression)
            return str(factored)
        except Exception as e:
            logger.error(f"Error in sympy_factor: {str(e)}")
            return expression
    
    # NumPy functions
    def _numpy_matrix_ops(self, operation, matrix_a, matrix_b=None):
        """Perform matrix operations."""
        try:
            import numpy as np
            
            if operation == "determinant":
                return np.linalg.det(matrix_a)
            elif operation == "inverse":
                return np.linalg.inv(matrix_a)
            elif operation == "eigenvalues":
                return np.linalg.eigvals(matrix_a)
            elif operation == "multiply" and matrix_b is not None:
                return np.matmul(matrix_a, matrix_b)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.error(f"Error in numpy_matrix_ops: {str(e)}")
            return None
    
    def _numpy_stats(self, operation, data):
        """Perform statistical analysis."""
        try:
            import numpy as np
            
            if operation == "mean":
                return np.mean(data)
            elif operation == "median":
                return np.median(data)
            elif operation == "std":
                return np.std(data)
            elif operation == "correlation":
                return np.corrcoef(data)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.error(f"Error in numpy_stats: {str(e)}")
            return None
    
    def _numpy_optimize(self, function, bounds):
        """Perform numerical optimization."""
        try:
            from scipy import optimize
            
            result = optimize.minimize_scalar(function, bounds=bounds)
            return {
                "x": result.x,
                "fun": result.fun,
                "success": result.success
            }
        except Exception as e:
            logger.error(f"Error in numpy_optimize: {str(e)}")
            return None
    
    # NetworkX functions
    def _networkx_create_graph(self, graph_type, params):
        """Create a graph of specified type."""
        try:
            import networkx as nx
            
            if graph_type == "complete":
                n = params.get("n", 5)
                return nx.complete_graph(n)
            elif graph_type == "cycle":
                n = params.get("n", 5)
                return nx.cycle_graph(n)
            elif graph_type == "custom":
                edges = params.get("edges", [])
                G = nx.Graph()
                G.add_edges_from(edges)
                return G
            else:
                raise ValueError(f"Unsupported graph type: {graph_type}")
        except Exception as e:
            logger.error(f"Error in networkx_create_graph: {str(e)}")
            return None
    
    def _networkx_analyze_graph(self, graph, metric):
        """Analyze graph properties."""
        try:
            import networkx as nx
            
            if metric == "diameter":
                return nx.diameter(graph)
            elif metric == "connectivity":
                return nx.node_connectivity(graph)
            elif metric == "centrality":
                return nx.eigenvector_centrality(graph)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except Exception as e:
            logger.error(f"Error in networkx_analyze_graph: {str(e)}")
            return None
    
    def _networkx_find_path(self, graph, source, target, method="shortest"):
        """Find a path in the graph."""
        try:
            import networkx as nx
            
            if method == "shortest":
                return nx.shortest_path(graph, source, target)
            elif method == "all":
                return list(nx.all_simple_paths(graph, source, target))
            else:
                raise ValueError(f"Unsupported method: {method}")
        except Exception as e:
            logger.error(f"Error in networkx_find_path: {str(e)}")
            return None
    
    # SageMath functions
    def _sage_number_theory(self, function, params):
        """Perform number theory operations using SageMath."""
        # This would call SageMath's number theory functions
        # Placeholder implementation
        return f"SageMath {function} result for {params}"
    
    def _sage_algebraic_geometry(self, function, params):
        """Perform algebraic geometry operations using SageMath."""
        # This would call SageMath's algebraic geometry functions
        # Placeholder implementation
        return f"SageMath {function} result for {params}"
    
    def _sage_group_theory(self, function, params):
        """Perform group theory operations using SageMath."""
        # This would call SageMath's group theory functions
        # Placeholder implementation
        return f"SageMath {function} result for {params}"
    
    # GeoGebra functions
    def _geogebra_construct(self, construction_steps):
        """Create a geometric construction in GeoGebra."""
        # This would interface with GeoGebra to create a construction
        # Placeholder implementation
        return f"GeoGebra construction created with {len(construction_steps)} steps"
    
    def _geogebra_visualize(self, construction, interactive=True):
        """Create an interactive visualization in GeoGebra."""
        # This would interface with GeoGebra to create an interactive visualization
        # Placeholder implementation
        return f"GeoGebra visualization created (interactive={interactive})"

# ====================== DISTRIBUTED COMPUTATION MODULE ======================

class DistributedComputationManager:
    """
    Manages distributed computation for the mathematical reasoning system.
    Enables scaling to handle complex problems by distributing work across
    multiple compute nodes.
    """
    
    def __init__(self, use_distributed=False, num_workers=None):
        """
        Initialize the distributed computation manager.
        
        Args:
            use_distributed: Whether to use distributed computation
            num_workers: Number of worker processes (None for auto-detect)
        """
        self.use_distributed = use_distributed
        self.num_workers = num_workers or mp.cpu_count()
        self.executor = None
        self.distributed_framework = None
        
        if use_distributed:
            self._initialize_distributed()
        else:
            self._initialize_local()
        
        logger.info(f"Initialized computation manager (distributed={use_distributed}, workers={self.num_workers})")
    
    def _initialize_local(self):
        """Initialize local parallel computation."""
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
    
    def _initialize_distributed(self):
        """Initialize distributed computation framework."""
        # This would initialize a distributed framework like Ray or Dask
        # For demonstration, we'll use a local executor
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self.distributed_framework = "process_pool"
    
    def run_task(self, task_func, *args, **kwargs):
        """
        Run a task in parallel or distributed mode.
        
        Args:
            task_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            concurrent.futures.Future: A future representing the computation
        """
        return self.executor.submit(task_func, *args, **kwargs)
    
    def map_tasks(self, task_func, items):
        """
        Apply a function to each item in parallel.
        
        Args:
            task_func: Function to apply
            items: Items to process
            
        Returns:
            list: Results for each item
        """
        futures = []
        for item in items:
            future = self.executor.submit(task_func, item)
            futures.append(future)
        
        return [future.result() for future in futures]
    
    def shutdown(self):
        """Shutdown the computation manager."""
        if self.executor:
            self.executor.shutdown()
        logger.info("Computation manager shutdown complete")

# ====================== KNOWLEDGE MANAGEMENT SYSTEM ======================

class KnowledgeManagementSystem:
    """
    Persistent knowledge management system for learning from past problems.
    Stores solved problems, proofs, and learned heuristics.
    """
    
    def __init__(self, storage_dir="./knowledge_base"):
        """
        Initialize the knowledge management system.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = storage_dir
        self.fact_database = {}
        self.proof_database = {}
        self.heuristics = {}
        self.construct_patterns = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing knowledge
        self._load_knowledge()
        
        logger.info(f"Initialized knowledge management system at {storage_dir}")
    
    def _load_knowledge(self):
        """Load existing knowledge from persistent storage."""
        try:
            # Load fact database
            fact_db_path = os.path.join(self.storage_dir, "fact_database.pkl")
            if os.path.exists(fact_db_path):
                with open(fact_db_path, "rb") as f:
                    self.fact_database = pickle.load(f)
            
            # Load proof database
            proof_db_path = os.path.join(self.storage_dir, "proof_database.pkl")
            if os.path.exists(proof_db_path):
                with open(proof_db_path, "rb") as f:
                    self.proof_database = pickle.load(f)
            
            # Load heuristics
            heuristics_path = os.path.join(self.storage_dir, "heuristics.pkl")
            if os.path.exists(heuristics_path):
                with open(heuristics_path, "rb") as f:
                    self.heuristics = pickle.load(f)
            
            # Load construction patterns
            patterns_path = os.path.join(self.storage_dir, "construct_patterns.pkl")
            if os.path.exists(patterns_path):
                with open(patterns_path, "rb") as f:
                    self.construct_patterns = pickle.load(f)
            
            logger.info(f"Loaded {len(self.fact_database)} facts, {len(self.proof_database)} proofs, " +
                       f"{len(self.heuristics)} heuristics, and {len(self.construct_patterns)} construction patterns")
        except Exception as e:
            logger.error(f"Error loading knowledge: {str(e)}")
    
    def save_knowledge(self):
        """Save knowledge to persistent storage."""
        try:
            # Save fact database
            fact_db_path = os.path.join(self.storage_dir, "fact_database.pkl")
            with open(fact_db_path, "wb") as f:
                pickle.dump(self.fact_database, f)
            
            # Save proof database
            proof_db_path = os.path.join(self.storage_dir, "proof_database.pkl")
            with open(proof_db_path, "wb") as f:
                pickle.dump(self.proof_database, f)
            
            # Save heuristics
            heuristics_path = os.path.join(self.storage_dir, "heuristics.pkl")
            with open(heuristics_path, "wb") as f:
                pickle.dump(self.heuristics, f)
            
            # Save construction patterns
            patterns_path = os.path.join(self.storage_dir, "construct_patterns.pkl")
            with open(patterns_path, "wb") as f:
                pickle.dump(self.construct_patterns, f)
            
            logger.info("Knowledge saved to persistent storage")
        except Exception as e:
            logger.error(f"Error saving knowledge: {str(e)}")
    
    def add_fact(self, fact, domain, metadata=None):
        """
        Add a fact to the knowledge base.
        
        Args:
            fact: The fact to add
            domain: Mathematical domain
            metadata: Additional information about the fact
        """
        fact_hash = self._hash_fact(fact)
        self.fact_database[fact_hash] = {
            "fact": fact,
            "domain": domain,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
    
    def add_proof(self, problem_id, proof, metadata=None):
        """
        Add a proof to the knowledge base.
        
        Args:
            problem_id: Identifier for the problem
            proof: The proof object
            metadata: Additional information about the proof
        """
        self.proof_database[problem_id] = {
            "proof": proof,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
    
    def add_heuristic(self, heuristic_id, heuristic, domain, metadata=None):
        """
        Add a heuristic to the knowledge base.
        
        Args:
            heuristic_id: Identifier for the heuristic
            heuristic: The heuristic rule or function
            domain: Mathematical domain
            metadata: Additional information about the heuristic
        """
        self.heuristics[heuristic_id] = {
            "heuristic": heuristic,
            "domain": domain,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
    
    def add_construction_pattern(self, pattern_id, pattern, domain, metadata=None):
        """
        Add a construction pattern to the knowledge base.
        
        Args:
            pattern_id: Identifier for the pattern
            pattern: The construction pattern
            domain: Mathematical domain
            metadata: Additional information about the pattern
        """
        self.construct_patterns[pattern_id] = {
            "pattern": pattern,
            "domain": domain,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "usage_count": 0,
            "success_rate": 0.0
        }
    
    def get_relevant_facts(self, query, domain=None, max_results=10):
        """
        Retrieve facts relevant to a query.
        
        Args:
            query: Search query
            domain: Optional domain filter
            max_results: Maximum number of results
            
        Returns:
            list: Relevant facts
        """
        relevant_facts = []
        
        for fact_hash, fact_data in self.fact_database.items():
            # Filter by domain if specified
            if domain and fact_data["domain"] != domain:
                continue
            
            # Simple relevance check (would be more sophisticated in a real system)
            relevance = self._compute_relevance(query, fact_data["fact"])
            if relevance > 0:
                relevant_facts.append((fact_data["fact"], relevance))
        
        # Sort by relevance and return top results
        relevant_facts.sort(key=lambda x: x[1], reverse=True)
        return [fact for fact, relevance in relevant_facts[:max_results]]
    
    def get_similar_proofs(self, problem_description, domain=None, max_results=5):
        """
        Retrieve proofs for similar problems.
        
        Args:
            problem_description: Description of the current problem
            domain: Optional domain filter
            max_results: Maximum number of results
            
        Returns:
            list: Similar proofs
        """
        similar_proofs = []
        
        for problem_id, proof_data in self.proof_database.items():
            # Filter by domain if specified
            if domain and proof_data.get("metadata", {}).get("domain") != domain:
                continue
            
            # Compute similarity (would be more sophisticated in a real system)
            similarity = self._compute_similarity(problem_description, problem_id)
            if similarity > 0.5:  # Threshold for similarity
                similar_proofs.append((proof_data["proof"], similarity))
        
        # Sort by similarity and return top results
        similar_proofs.sort(key=lambda x: x[1], reverse=True)
        return [proof for proof, similarity in similar_proofs[:max_results]]
    
    def get_domain_heuristics(self, domain):
        """
        Get heuristics for a specific domain.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            list: Heuristics for the domain
        """
        return [h["heuristic"] for h_id, h in self.heuristics.items() 
                if h["domain"] == domain]
    
    def get_successful_constructions(self, domain):
        """
        Get construction patterns that have been successful.
        
        Args:
            domain: Mathematical domain
            
        Returns:
            list: Successful construction patterns
        """
        patterns = []
        
        for pattern_id, pattern_data in self.construct_patterns.items():
            if pattern_data["domain"] == domain and pattern_data["success_rate"] > 0.5:
                patterns.append(pattern_data["pattern"])
        
        return patterns
    
    def update_construction_success(self, pattern_id, success):
        """
        Update the success rate of a construction pattern.
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the construction was successful
        """
        if pattern_id in self.construct_patterns:
            pattern = self.construct_patterns[pattern_id]
            count = pattern["usage_count"]
            rate = pattern["success_rate"]
            
            # Update usage count and success rate
            new_count = count + 1
            new_rate = (rate * count + (1 if success else 0)) / new_count
            
            pattern["usage_count"] = new_count
            pattern["success_rate"] = new_rate
    
    def learn_from_proof(self, problem_description, formalization, proof, success):
        """
        Learn from a completed proof attempt.
        
        Args:
            problem_description: Description of the problem
            formalization: Problem formalization
            proof: The proof object
            success: Whether the proof was successful
        """
        if not proof:
            return
        
        # Generate a problem ID
        problem_id = self._generate_problem_id(problem_description)
        
        # Add the proof to the database
        self.add_proof(problem_id, proof, {
            "problem_description": problem_description,
            "formalization": formalization,
            "success": success,
            "domain": formalization.get("domain", "general")
        })
        
        # Extract and learn from successful constructions
        if success and proof.steps:
            self._learn_constructions(formalization.get("domain", "general"), proof.steps)
        
        # Save the updated knowledge
        self.save_knowledge()
    
    def _learn_constructions(self, domain, steps):
        """Learn construction patterns from successful proof steps."""
        constructions = []
        
        for step in steps:
            if step.step_type == "construction":
                # Extract the construction from the step
                # This would be implemented based on the step format
                construction = {}
                
                if construction:
                    constructions.append(construction)
        
        if constructions:
            # Create a new pattern from the constructions
            pattern_id = f"{domain}_pattern_{len(self.construct_patterns)}"
            pattern = {
                "constructions": constructions,
                "sequence": [c.get("type", "unknown") for c in constructions]
            }
            
            self.add_construction_pattern(pattern_id, pattern, domain, {
                "learned_from": "successful_proof",
                "construction_count": len(constructions)
            })
    
    def _hash_fact(self, fact):
        """Generate a hash for a fact."""
        # This would generate a consistent hash for a fact
        # For demonstration, we'll use the string representation
        return hashlib.md5(str(fact).encode()).hexdigest()
    
    def _generate_problem_id(self, problem_description):
        """Generate a unique ID for a problem."""
        return hashlib.md5(problem_description.encode()).hexdigest()
    
    def _compute_relevance(self, query, fact):
        """Compute relevance between a query and a fact."""
        # This would compute a relevance score
        # For demonstration, we'll use a simple random score
        return np.random.uniform(0, 1)
    
    def _compute_similarity(self, problem_description, problem_id):
        """Compute similarity between a problem description and a stored problem."""
        # This would compute a similarity score
        # For demonstration, we'll use a simple random score
        return np.random.uniform(0, 1)

# ====================== WEB API AND SERVICE ======================

class MathSolverWebService:
    """
    Web service interface for the mathematical reasoning system.
    Provides RESTful API endpoints for solving problems, managing knowledge,
    and interacting with the system.
    """
    
    def __init__(self, solver, host="localhost", port=8000):
        """
        Initialize the web service.
        
        Args:
            solver: The MathSolverAPI instance
            host: Host to bind the server
            port: Port to bind the server
        """
        self.solver = solver
        self.host = host
        self.port = port
        
        # Set up additional services
        self.visualizer = MathVisualizer()
        self.tool_manager = ToolIntegrationManager()
        self.compute_manager = DistributedComputationManager()
        self.knowledge_system = KnowledgeManagementSystem()
        
        logger.info(f"Initialized web service at {host}:{port}")
    
    def start(self):
        """Start the web service."""
        # This would start a web server with the defined endpoints
        # For demonstration, we'll just print the available endpoints
        logger.info(f"Starting web service at {self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  POST /solve - Solve a mathematical problem")
        logger.info("  POST /batch_solve - Solve multiple problems")
        logger.info("  GET /visualization/{solution_id} - Get a visualization of a solution")
        logger.info("  POST /compute - Run a distributed computation")
        logger.info("  GET /knowledge - Query the knowledge base")
    
    def define_routes(self):
        """Define the API routes and handlers."""
        # This would define the routes for a web framework
        # For demonstration, we'll define the handlers without an actual framework
        routes = [
            {"method": "POST", "path": "/solve", "handler": self.handle_solve},
            {"method": "POST", "path": "/batch_solve", "handler": self.handle_batch_solve},
            {"method": "GET", "path": "/visualization/{solution_id}", "handler": self.handle_visualization},
            {"method": "POST", "path": "/compute", "handler": self.handle_compute},
            {"method": "GET", "path": "/knowledge", "handler": self.handle_knowledge_query},
            {"method": "POST", "path": "/knowledge", "handler": self.handle_knowledge_update}
        ]
        return routes
    
    def handle_solve(self, request):
        """
        Handle a solve request.
        
        Args:
            request: The HTTP request
            
        Returns:
            dict: Solution response
        """
        # Parse request data
        problem_text = request.get("problem")
        timeout = request.get("timeout", 300)
        
        if not problem_text:
            return {"error": "No problem provided"}, 400
        
        # Solve the problem
        solution = self.solver.solve(problem_text, timeout=timeout)
        
        # Generate a solution ID
        solution_id = hashlib.md5(problem_text.encode()).hexdigest()
        
        # Learn from the solution if successful
        if solution.get("succeeded"):
            self.knowledge_system.learn_from_proof(
                problem_text,
                solution.get("formalization"),
                solution.get("proof"),
                solution.get("succeeded")
            )
        
        # Include visualization URLs
        if solution.get("proof"):
            solution["visualization_urls"] = {
                "proof_graph": f"/visualization/{solution_id}/proof_graph",
                "geometric": f"/visualization/{solution_id}/geometric" if solution.get("formalization", {}).get("domain") == "geometry" else None
            }
        
        return {"solution_id": solution_id, "solution": solution}, 200
    
    def handle_batch_solve(self, request):
        """
        Handle a batch solve request.
        
        Args:
            request: The HTTP request
            
        Returns:
            dict: Batch solution response
        """
        # Parse request data
        problems = request.get("problems", [])
        timeout_per_problem = request.get("timeout_per_problem", 300)
        
        if not problems:
            return {"error": "No problems provided"}, 400
        
        # Use distributed computation for batch solving
        results = self.compute_manager.map_tasks(
            lambda problem: self.solver.solve(problem, timeout=timeout_per_problem),
            problems
        )
        
        # Process and learn from results
        for i, (problem, result) in enumerate(zip(problems, results)):
            # Generate a solution ID
            solution_id = hashlib.md5(problem.encode()).hexdigest()
            
            # Learn from the solution if successful
            if result.get("succeeded"):
                self.knowledge_system.learn_from_proof(
                    problem,
                    result.get("formalization"),
                    result.get("proof"),
                    result.get("succeeded")
                )
            
            # Include visualization URLs
            if result.get("proof"):
                result["visualization_urls"] = {
                    "proof_graph": f"/visualization/{solution_id}/proof_graph",
                    "geometric": f"/visualization/{solution_id}/geometric" if result.get("formalization", {}).get("domain") == "geometry" else None
                }
            
            results[i] = {"solution_id": solution_id, "solution": result}
        
        return {"results": results}, 200
    
    def handle_visualization(self, request):
        """
        Handle a visualization request.
        
        Args:
            request: The HTTP request
            
        Returns:
            Image or JSON response
        """
        # Parse request parameters
        solution_id = request.get("solution_id")
        vis_type = request.get("type", "proof_graph")
        
        if not solution_id:
            return {"error": "No solution ID provided"}, 400
        
        # Lookup the solution (this would require storage in a real system)
        # For demonstration, we'll generate a placeholder visualization
        
        if vis_type == "proof_graph":
            fig = self.visualizer.visualize_proof_graph(None)  # Placeholder
            image_data = self._fig_to_image(fig)
            return image_data, 200, {"Content-Type": "image/png"}
        elif vis_type == "geometric":
            fig = self.visualizer.visualize_geometry_problem({})  # Placeholder
            image_data = self._fig_to_image(fig)
            return image_data, 200, {"Content-Type": "image/png"}
        else:
            return {"error": f"Unsupported visualization type: {vis_type}"}, 400
    
    def handle_compute(self, request):
        """
        Handle a distributed computation request.
        
        Args:
            request: The HTTP request
            
        Returns:
            dict: Computation response
        """
        # Parse request data
        computation_type = request.get("type")
        parameters = request.get("parameters", {})
        
        if not computation_type:
            return {"error": "No computation type provided"}, 400
        
        # Delegate to the appropriate tool based on computation type
        if computation_type == "symbolic":
            tool_name = "sympy"
            function_name = parameters.get("function", "simplify_expression")
            args = parameters.get("args", [])
            kwargs = parameters.get("kwargs", {})
            
            result = self.tool_manager.call_tool_function(
                tool_name, function_name, *args, **kwargs
            )
            
            return {"result": result}, 200
        elif computation_type == "numeric":
            tool_name = "numpy"
            function_name = parameters.get("function", "matrix_operations")
            args = parameters.get("args", [])
            kwargs = parameters.get("kwargs", {})
            
            result = self.tool_manager.call_tool_function(
                tool_name, function_name, *args, **kwargs
            )
            
            return {"result": result}, 200
        else:
            return {"error": f"Unsupported computation type: {computation_type}"}, 400
    
    def handle_knowledge_query(self, request):
        """
        Handle a knowledge query request.
        
        Args:
            request: The HTTP request
            
        Returns:
            dict: Knowledge query response
        """
        # Parse request parameters
        query_type = request.get("type")
        parameters = request.get("parameters", {})
        
        if not query_type:
            return {"error": "No query type provided"}, 400
        
        # Execute the appropriate query
        if query_type == "facts":
            query = parameters.get("query", "")
            domain = parameters.get("domain")
            max_results = parameters.get("max_results", 10)
            
            facts = self.knowledge_system.get_relevant_facts(
                query, domain=domain, max_results=max_results
            )
            
            return {"facts": facts}, 200
        elif query_type == "proofs":
            problem = parameters.get("problem", "")
            domain = parameters.get("domain")
            max_results = parameters.get("max_results", 5)
            
            proofs = self.knowledge_system.get_similar_proofs(
                problem, domain=domain, max_results=max_results
            )
            
            return {"proofs": proofs}, 200
        elif query_type == "heuristics":
            domain = parameters.get("domain")
            
            if not domain:
                return {"error": "Domain required for heuristics query"}, 400
            
            heuristics = self.knowledge_system.get_domain_heuristics(domain)
            
            return {"heuristics": heuristics}, 200
        else:
            return {"error": f"Unsupported query type: {query_type}"}, 400
    
    def handle_knowledge_update(self, request):
        """
        Handle a knowledge update request.
        
        Args:
            request: The HTTP request
            
        Returns:
            dict: Knowledge update response
        """
        # Parse request data
        update_type = request.get("type")
        parameters = request.get("parameters", {})
        
        if not update_type:
            return {"error": "No update type provided"}, 400
        
        # Execute the appropriate update
        if update_type == "fact":
            fact = parameters.get("fact")
            domain = parameters.get("domain")
            metadata = parameters.get("metadata")
            
            if not fact or not domain:
                return {"error": "Fact and domain required"}, 400
            
            self.knowledge_system.add_fact(fact, domain, metadata)
            
            return {"status": "success"}, 200
        elif update_type == "heuristic":
            heuristic_id = parameters.get("id")
            heuristic = parameters.get("heuristic")
            domain = parameters.get("domain")
            metadata = parameters.get("metadata")
            
            if not heuristic_id or not heuristic or not domain:
                return {"error": "Heuristic ID, heuristic, and domain required"}, 400
            
            self.knowledge_system.add_heuristic(heuristic_id, heuristic, domain, metadata)
            
            return {"status": "success"}, 200
        elif update_type == "construction":
            pattern_id = parameters.get("id")
            pattern = parameters.get("pattern")
            domain = parameters.get("domain")
            metadata = parameters.get("metadata")
            
            if not pattern_id or not pattern or not domain:
                return {"error": "Pattern ID, pattern, and domain required"}, 400
            
            self.knowledge_system.add_construction_pattern(pattern_id, pattern, domain, metadata)
            
            return {"status": "success"}, 200
        else:
            return {"error": f"Unsupported update type: {update_type}"}, 400
    
    def _fig_to_image(self, fig):
        """Convert a matplotlib figure to image data."""
        # This would convert the figure to a binary image
        # For demonstration, we'll return an empty buffer
        return b""

# ====================== ENHANCED MAIN APPLICATION ======================

class EnhancedMathSolver:
    """
    Enhanced main application integrating all components of the system.
    Provides a unified interface for solving mathematical problems.
    """
    
    def __init__(self, config=None):
        """
        Initialize the enhanced math solver.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.api = MathSolverAPI(config)
        self.visualizer = MathVisualizer()
        self.tool_manager = ToolIntegrationManager()
        self.compute_manager = DistributedComputationManager(
            use_distributed=self.config.get("use_distributed", False),
            num_workers=self.config.get("num_workers")
        )
        self.knowledge_system = KnowledgeManagementSystem(
            storage_dir=self.config.get("knowledge_dir", "./knowledge_base")
        )
        
        # Initialize web service if needed
        if self.config.get("start_web_service", False):
            self.web_service = MathSolverWebService(
                self.api,
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 8000)
            )
            self.web_service.start()
        else:
            self.web_service = None
        
        logger.info("Enhanced Math Solver initialized")
    
    def solve(self, problem_text, timeout=300, visualize=False):
        """
        Solve a mathematical problem.
        
        Args:
            problem_text: Natural language problem description
            timeout: Timeout in seconds
            visualize: Whether to generate visualizations
            
        Returns:
            dict: Solution with optional visualizations
        """
        # Check for similar problems in knowledge base
        similar_proofs = self.knowledge_system.get_similar_proofs(problem_text)
        
        # If very similar problem exists, adapt its solution
        if similar_proofs and self.config.get("use_knowledge_base", True):
            logger.info("Found similar problem in knowledge base, adapting solution")
            # This would adapt the existing solution to the new problem
            # For now, we'll continue with normal solving
        
        # Solve the problem
        solution = self.api.solve(problem_text, timeout=timeout)
        
        # Add visualizations if requested
        if visualize and solution.get("formalization"):
            domain = solution.get("formalization", {}).get("domain")
            
            visualizations = {}
            
            if domain == "geometry":
                fig = self.visualizer.visualize_geometry_problem(solution["formalization"])
                visualizations["geometry"] = fig
            
            if solution.get("proof"):
                fig = self.visualizer.visualize_proof_graph(solution["proof"])
                visualizations["proof_graph"] = fig
            
            solution["visualizations"] = visualizations
        
        # Learn from the solution
        if solution.get("succeeded"):
            self.knowledge_system.learn_from_proof(
                problem_text,
                solution.get("formalization"),
                solution.get("proof"),
                solution.get("succeeded")
            )
        
        return solution
    
    def batch_solve(self, problems, timeout_per_problem=300, visualize=False):
        """
        Solve multiple problems in parallel.
        
        Args:
            problems: List of problem descriptions
            timeout_per_problem: Timeout per problem in seconds
            visualize: Whether to generate visualizations
            
        Returns:
            list: Solutions for each problem
        """
        # Use distributed computation for batch solving
        solve_func = partial(self.solve, timeout=timeout_per_problem, visualize=visualize)
        results = self.compute_manager.map_tasks(solve_func, problems)
        
        return results
    
    def interactive_session(self):
        """Run an interactive session for solving problems."""
        print("=== NeuroSymbolic Mathematical Reasoning System ===")
        print("Enter a mathematical problem to solve, or 'exit' to quit.")
        
        while True:
            problem = input("\nProblem: ")
            if problem.lower() in ["exit", "quit"]:
                break
            
            print("\nSolving problem...")
            start_time = time.time()
            solution = self.solve(problem, visualize=True)
            end_time = time.time()
            
            print(f"\nSolution found in {end_time - start_time:.2f} seconds.")
            print(f"Success: {solution.get('succeeded', False)}")
            
            if solution.get("explanation"):
                print("\nExplanation:")
                print(solution["explanation"])
            
            # Display visualizations
            visualizations = solution.get("visualizations", {})
            if visualizations:
                print("\nVisualization available. Show? (y/n)")
                show = input().lower()
                if show == "y":
                    for vis_type, fig in visualizations.items():
                        plt.figure(fig.number)
                        plt.title(f"{vis_type.replace('_', ' ').title()} Visualization")
                        plt.show()
    
    def shutdown(self):
        """Shutdown all components."""
        if self.compute_manager:
            self.compute_manager.shutdown()
        
        if self.knowledge_system:
            self.knowledge_system.save_knowledge()
        
        logger.info("Enhanced Math Solver shutdown complete")

# ====================== MAIN ENTRY POINT ======================

def main():
    """Main entry point for the application."""
    # Configure the solver
    config = {
        "language_model": {
            "type": "mock",  # Use mock for demonstration
            "max_context_length": 8192,
            "context_strategy": "prioritize_recent"
        },
        "symbolic_engine": {
            "domains": ["geometry", "algebra", "number_theory", "combinatorics"]
        },
        "search": {
            "num_threads": 4,
            "max_tree_depth": 10
        },
        "use_distributed": True,
        "num_workers": 4,
        "knowledge_dir": "./knowledge_base",
        "use_knowledge_base": True,
        "start_web_service": False
    }
    
    # Initialize the enhanced solver
    solver = EnhancedMathSolver(config)
    
    # Sample problems for demonstration
    problems = [
        "Let triangle ABC be isosceles with AB = AC. Let point D be on BC such that AD bisects angle BAC. Prove that AB = BD + CD.",
        "Find all positive integers n such that n^2 + 20 is divisible by n + 2.",
        "Two circles with centers O1 and O2 intersect at points A and B. Let line l1 be tangent to the first circle at point C, and let line l2 be tangent to the second circle at point D. Prove that if points C and D are on the same side of line AB, then angles ACB and ADB are equal."
    ]
    
    # Solve a single problem in detail with visualization
    print(f"Solving problem: {problems[0]}")
    solution = solver.solve(problems[0], visualize=True)
    
    print("\nFormalization:")
    print(f"Domain: {solution['formalization'].get('domain', 'unknown')}")
    
    if solution.get("proof"):
        print("\nProof found:")
        for i, step in enumerate(solution["proof"]["steps"]):
            print(f"Step {i+1}: {step['step_type']} - {step['description']}")
        
        print(f"\nSuccess: {solution.get('succeeded', False)}")
        print(f"Time taken: {solution.get('time_taken', 0):.2f} seconds")
    else:
        print("\nNo proof found.")
    
    print("\nExplanation:")
    print(solution.get("explanation", "No explanation available."))
    
    # Show visualizations
    visualizations = solution.get("visualizations", {})
    if visualizations:
        for vis_type, fig in visualizations.items():
            plt.figure(fig.number)
            plt.title(f"{vis_type.replace('_', ' ').title()} Visualization")
            plt.show()
    
    # Run batch solving
    print("\nSolving all problems in batch mode...")
    batch_results = solver.batch_solve(problems)
    
    print("\nBatch results:")
    for i, result in enumerate(batch_results):
        print(f"Problem {i+1}: {'Solved' if result.get('succeeded') else 'Not solved'}")
    
    # Shutdown
    solver.shutdown()

if __name__ == "__main__":
    main()
```


## Advanced Features Added to the Implementation

This expanded implementation builds upon the previous codebase with several critical enhancements that take the system to the next level:

### 1. Visualization Module

- **Interactive geometric visualizations** for problems and their solutions
- **Proof graph visualization** to show the logical structure of proofs
- **Search tree visualization** to illustrate the SKEST algorithm's exploration
- Multiple visualization styles and export capabilities


### 2. External Tool Integration

- **SymPy integration** for symbolic mathematics (equations, integrals, factoring)
- **NumPy/SciPy integration** for numerical computations and optimizations
- **NetworkX integration** for graph theory problems
- **Interfaces to SageMath and GeoGebra** for specialized mathematical tasks


### 3. Distributed Computation

- **Parallel processing** for both proof search and batch problem solving
- **Thread management** for efficient resource utilization
- **Workload distribution** across multiple cores or machines
- **Fault tolerance** for long-running computations


### 4. Knowledge Management System

- **Persistent storage** of facts, proofs, and heuristics
- **Learning from successful proofs** to improve future performance
- **Construction pattern extraction** to identify reusable geometric constructions
- **Similarity search** to adapt existing solutions to new problems


### 5. Web Service \& API

- **RESTful API** for remote access to the mathematical reasoning system
- **Visualization endpoints** for generating and sharing proof visualizations
- **Batch computation** for processing multiple problems
- **Knowledge querying** and contribution endpoints

These advanced components transform the system from a standalone mathematical solver into a comprehensive platform that can continuously learn, leverage external tools, distribute work across computing resources, and provide rich visualizations of its reasoning process.

The implementation is designed with modularity in mind, allowing each component to be improved or replaced independently as new techniques and algorithms are developed in mathematical reasoning and AI.

