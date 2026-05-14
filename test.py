import numpy as np

def add_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(1, 6)
    return start_value + random_value, random_value, "addition"

def subtract_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(1, 6)
    return start_value - random_value, random_value, "subtraction"

def multiply_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(2, 6)
    return start_value * random_value, random_value, "multiplication"

def divide_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(2, 6)
    return start_value / random_value, random_value, "division"

def power_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(2, 4)
    return start_value ** random_value, random_value, "power"

def root_and_value(start_value: int, random_value: int = None):
    if random_value is None:
        random_value = np.random.randint(2, 4)
    return start_value ** (1/random_value), random_value, "root"

def get_opposite_operation(operation_name: str):
    if operation_name == "addition":
        return subtract_and_value
    elif operation_name == "subtraction":
        return add_and_value
    elif operation_name == "multiplication":
        return divide_and_value
    elif operation_name == "division":
        return multiply_and_value
    elif operation_name == "power":
        return root_and_value

def get_opposite_operation_str(operation_name: str):
    if operation_name == "addition":
        return "subtraction"
    elif operation_name == "subtraction":
        return "addition"
    elif operation_name == "multiplication":
        return "division"
    elif operation_name == "division":
        return "multiplication"
    elif operation_name == "power":
        return "root"

def get_random_operation_and_value():
    operations = [add_and_value, subtract_and_value, multiply_and_value, divide_and_value, root_and_value]
    operation = np.random.choice(operations, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    return operation

def apply_random_operations(start_value: int, num_operations: int):
    value = start_value
    path = []
    for _ in range(num_operations):
        while True:
            operation = get_random_operation_and_value()
            new_value, random_value, operation_name = operation(value)
            if new_value > 0 and new_value < 1e6 and int(new_value) == new_value and new_value != start_value:
                value = new_value
                break
        path.append((operation_name, random_value))
    return value, path

def generate_options(path):
    options = [(get_opposite_operation_str(operation_name), random_value) for operation_name, random_value in reversed(path)]
    for _ in range(2):
        while True:
            operation = get_random_operation_and_value()
            if operation == root_and_value:
                operation = power_and_value
            _, random_value, operation_name = operation(1)
            if (operation_name, random_value) not in options:
                break
        options.append((operation_name, random_value))
    np.random.shuffle(options)
    return options

def print_path(path, start_value, final_value, print_solution=True):
    if print_solution:
        print(f"Start value: {final_value}")
        print("Operations:")
        for operation_name, random_value in reversed(path):
            opposite_operation = get_opposite_operation(operation_name)
            final_value, _, op_str = opposite_operation(final_value, random_value)
            print(f"  {op_str}: {random_value:.2f} -> {final_value:.2f}")
        print(f"Final value: {start_value}")
    else:
        print(f"Start value: {start_value}")
        print("Path of operations:")
        for operation_name, random_value in path:
            print(f"  {operation_name}: {random_value:.2f}")
        print(f"Final value: {start_value}")
        

def print_options(options):
    print("Options to apply:")
    for i, (operation_name, random_value) in enumerate(options):
        print(f"{i+1}. {operation_name} with {int(random_value)}")

def play_round(min_start_value = 3, max_start_value = 11, num_operations = 3):
    start_value = np.random.randint(min_start_value, max_start_value)
    final_value, path = apply_random_operations(start_value, num_operations)
    
    # print_path(path, start_value, final_value, True)
        
    options = generate_options(path)
     
    print("-"*40)
    print(f"\nStart value: {final_value}")
    print(f"\nGoal value: {start_value}\n")
    current_value = final_value
    while True:
        print_options(options)
        print("")
        max_choice = len(options)
        choice = int(input(f"Choose an option to apply (1-{max_choice}): "))
        if choice < 1 or choice > max_choice:
            print(f"Invalid choice. Please choose a number between 1 and {max_choice}.")
            continue
        operation_name, random_value = options[choice - 1]
        if operation_name == "addition":
            current_value += random_value
        elif operation_name == "subtraction":
            current_value -= random_value
        elif operation_name == "multiplication":
            current_value *= random_value
        elif operation_name == "division":
            current_value /= random_value
        elif operation_name == "power":
            current_value **= random_value
            
        options.pop(choice - 1)
            
        print("")
        print(f"Current value: {current_value:.2f}")
        if current_value == start_value:
            print("Victory!")
            break
        if len(options) == 0:
            print("No more options left. Hahaha! xD")
            break
        
    print("")
    print("Solution:")
    print_path(path, start_value, final_value)
    print("")
    print("-"*40)

if __name__ == "__main__":
    num_operations = 3
    seed = 2123
    np.random.seed(seed)
    while True:
        play_round(num_operations=num_operations)
        choice = input(f"Play again? (y/n): ")
        if choice.lower() != "y":
            break