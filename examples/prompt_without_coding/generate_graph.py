from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator, ValidationError

class Node:
    def __init__(self, name, content, dependencies):
        self.name = name
        self.content = content
        self.dependencies = dependencies

class DAGBuilder:
    def __init__(self):
        self.dag = {}

    def add_node(self):
        name = prompt('Enter node name: ')
        if name in self.dag:
            print(f"Node {name} already exists.")
            return
        content = prompt('Enter node content: ')
        dependencies = prompt('Enter node dependencies (comma separated): ').split(',')
        print(dependencies)
        for dep in dependencies:
            if dep not in self.dag:
                print(f"Dependency {dep} does not exist.")
                return
        self.dag[name] = Node(name, content, dependencies)
        self.list_nodes()

    def list_nodes(self):
        for name, node in self.dag.items():
            print(f"Node Name: {name}, Node Content: {node.content}, Node Dependencies: {', '.join(node.dependencies)}")

if __name__ == '__main__':
    builder = DAGBuilder()
    while True:
        builder.list_nodes()
        print("\n1. Add Node\n2. Quit")
        choice = prompt('Enter your choice: ')
        if choice == '1':
            builder.add_node()
        elif choice == '2':
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")