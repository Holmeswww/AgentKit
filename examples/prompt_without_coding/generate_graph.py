import json
import os
from colorama import Fore, Style, init
try:
    import agentkit
    from agentkit import Graph, BaseNode
    import agentkit.llm_api
except ImportError:
    print("Please install the agentkit package.")
    print("Run: pip install agentkit-llm")
    exit(1)

class DAG:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def add_node(self, node_name, content):
        if node_name in self.nodes:
            return False, "Node already exists."
        self.nodes[node_name] = content
        self.edges[node_name] = []
        return True, f"Node '{node_name}' added."
    
    def update_node_content(self, node_name, new_content):
        if node_name not in self.nodes:
            return False, "Node does not exist."
        self.nodes[node_name] = new_content
        return True, f"Node '{node_name}' prompt updated."
    
    def edit_dependencies(self, node_name, new_dependencies):
        if node_name not in self.nodes:
            return False, "Node does not exist."
        new_dependencies = set(new_dependencies)
        existing_dependencies = set(self.edges[node_name])
        if new_dependencies == existing_dependencies:
            return True, "No change in dependencies."
        
        # Remove old dependencies not in new dependencies
        for dep in existing_dependencies - new_dependencies:
            self.edges[node_name].remove(dep)
        
        # Add new dependencies not in existing dependencies
        errors = []
        for dep in new_dependencies - existing_dependencies:
            if dep not in self.nodes:
                errors.append(f"'{dep}' does not exist.")
                continue
            if self.is_cyclic(dep, node_name):
                errors.append(f"Adding '{dep}' would create a cycle.")
                continue
            self.edges[node_name].append(dep)
        
        if errors:
            return False, " ".join(errors)
        return True, f"Dependencies of '{node_name}' updated successfully."
    
    def add_edge(self, dependent_node, target_node):
        if dependent_node not in self.nodes or target_node not in self.nodes:
            return False, "Both nodes must exist."
        if dependent_node in self.edges[target_node]:
            return False, "Dependency already exists."
        if self.is_cyclic(dependent_node, target_node):
            return False, "This dependency would create a cycle."
        self.edges[target_node].append(dependent_node)
        return True, f"Dependency from '{dependent_node}' to '{target_node}' added."
    
    def is_cyclic(self, dependent_node, target_node):
        visited = set()
        stack = [dependent_node]
        while stack:
            node = stack.pop()
            if node == target_node:
                return True
            if node not in visited:
                visited.add(node)
                for predecessor in self.edges[node]:
                    stack.append(predecessor)
        return False

    def display(self):
        if not self.nodes:
            print("The graph is currently empty.")
        else:
            for node in self.nodes:
                print(f"Node: {node}, Prompt: {self.nodes[node]}, Dependencies: {self.edges[node]}")

    def save_graph(self, filename):
        graph_data = {
            'nodes': self.nodes,
            'edges': self.edges
        }
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        return f"Graph saved successfully to {filename}"

    def load_graph(self, filename):
        try:
            with open(filename, 'r') as f:
                graph_data = json.load(f)
            self.nodes = graph_data['nodes']
            self.edges = graph_data['edges']
            return True, f"Graph loaded successfully from {filename}"
        except FileNotFoundError:
            return False, f"No such file: {filename}"
        except json.JSONDecodeError:
            return False, "Error decoding JSON data."
        except Exception as e:
            return False, f"An error occurred: {e}"

# Initialize Colorama
init(autoreset=True)

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def press_enter_to_continue():
    """Prompts the user to press Enter to continue."""
    input(f"\n{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")

def main():
    dag = DAG()
    clear_screen()
    print(Fore.GREEN + "Welcome to the DAG Manager!")
    print(Fore.GREEN + "---------------------------\n")
    action = ''

    while True:
        clear_screen()
        print(Fore.YELLOW + "DAG Manager Dashboard")
        print(Fore.YELLOW + "---------------------\n")
        print(Fore.LIGHTBLUE_EX + "Current DAG Nodes and their details:")
        dag.display()
        print("\nOptions:")
        print(Fore.LIGHTGREEN_EX + " [A] Add node")
        print(Fore.LIGHTGREEN_EX + " [E] Edit node")
        print(Fore.LIGHTGREEN_EX + " [L] Load graph")
        print(Fore.LIGHTGREEN_EX + " [S] Save graph")
        print(Fore.LIGHTGREEN_EX + " [R] Evaluate graph")
        print(Fore.LIGHTGREEN_EX + " [X] Exit")
        
        action = input(Fore.CYAN + "\nChoose an action (A, E, L, S, R, X): ").lower().strip()

        if action == 'l':
            filename = input(Fore.MAGENTA + "Enter filename to load the graph from: ").strip()
            success, message = dag.load_graph(filename)
            print(Fore.RED if not success else Fore.GREEN, message)
        elif action == 's':
            filename = input(Fore.MAGENTA + "Enter filename to save the graph: ").strip()
            print(Fore.GREEN + dag.save_graph(filename))
        elif action == 'a':
            node_name = input(Fore.MAGENTA + "Enter new node name: ").strip()
            node_content = input(Fore.MAGENTA + "Enter node prompt: ").strip()
            result, msg = dag.add_node(node_name, node_content)
            print(Fore.RED if not result else Fore.GREEN, msg)
            if result:
                dependencies = input(Fore.MAGENTA + "Enter dependencies (comma-separated, blank to finish): ").strip()
                if dependencies:
                    dependencies = dependencies.split(',')
                    result, msg = dag.edit_dependencies(node_name, dependencies)
                    print(Fore.RED if not result else Fore.GREEN, msg)
        elif action == 'e':
            node_name = input(Fore.MAGENTA + "Enter the node name to edit: ").strip()
            if node_name not in dag.nodes:
                print(Fore.RED + "Node does not exist.")
                press_enter_to_continue()
                continue
            new_content = input(Fore.MAGENTA + f"Enter new prompt for '{node_name}' (current: {dag.nodes[node_name]}): ").strip()
            result, msg = dag.update_node_content(node_name, new_content)
            print(Fore.RED if not result else Fore.GREEN, msg)
            new_dependencies = input(Fore.MAGENTA + "Enter new dependencies (comma-separated, blank for no change): ").strip()
            if new_dependencies:
                new_dependencies = new_dependencies.split(',')
                result, msg = dag.edit_dependencies(node_name, new_dependencies)
                print(Fore.RED if not result else Fore.GREEN, msg)
        elif action == 'x':
            if input(Fore.RED + "Are you sure you want to exit? (y/n): ").lower().strip() == 'y':
                return
        elif action == 'r':
            break
        else:
            print(Fore.RED + "Invalid option, please try again.")
        
        press_enter_to_continue()
    
    llm_name = input(Fore.MAGENTA + "Enter the name of the LLM to generate the graph (e.g., gpt-4-turbo-preview): ").strip()
    LLM_API_FUNCTION = agentkit.llm_api.get_query(llm_name)
    graph = Graph()

    edges = []
    for node_name, node_content in dag.nodes.items():
        node = BaseNode(node_name, node_content, graph, LLM_API_FUNCTION, agentkit.compose_prompt.BaseComposePrompt())
        graph.add_node(node)

        for dep in dag.edges[node_name]:
            edges.append((dep, node_name))
    
    for dep, target in edges:
        graph.add_edge(dep, target)
    
    print(Fore.YELLOW + "Evaluating the graph...")
    result = graph.evaluate()

    for key, value in result.items():
        print(f"{key}: {value}")



if __name__ == "__main__":
    main()