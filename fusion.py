import onnx
from onnx import helper
from onnx import TensorProto

# Find add node following the current node
def find_following_add_node(graph, mm_node):
    for node in graph.node:
        if node.op_type == 'Add' and mm_node.output[0] in node.input:
            #print(f"Found Matmul node {mm_node.name} followed by Add node {node.name}")
            return node
    return None

# Find reshape node following the current node
def find_following_reshape_node(graph, gemm_node):
    for node in graph.node:
        if node.op_type == 'Reshape' and gemm_node.output[0] in node.input:
            #print(f"Found GEMM node {gemm_node.name} followed by Reshape node {node.name}")
            return node
    return None

# Find reshape node preceeding the current node
def find_previous_reshape_node(graph, gemm_node):
    for node in graph.node:
        if node.op_type == 'Reshape' and gemm_node.op_type == 'Gemm' and node.output[0] in gemm_node.input:
            #print(f"Found Reshape node {node.name} followed by GEMM node {gemm_node.name}")
            return node
    return None

# Find constant node associated with the current reshape node
def find_associated_constant_node(graph, reshape_node):
    # The second input of the Reshape node is the output of the Constant node
    shape_input_name = reshape_node.input[1]

    # Find the Constant node that outputs to this input
    for node in graph.node:
        if node.op_type == 'Constant' and node.output[0] == shape_input_name:
            return node

    # If no such node is found, return None
    return None

# Find the following node
def find_following_node(graph, node):
    # The output of the given node
    node_output = node.output[0]

    # Find the node that uses this output as an input
    for next_node in graph.node:
        if node_output in next_node.input:
            return next_node

    # If no such node is found, return None
    return None

# Find the output nodes to the current node
def find_output_nodes(graph, node):
    output_nodes = []
    node_output = node.output[0]

    for next_node in graph.node:
        if node_output in next_node.input:
            output_nodes.append(next_node)

    return output_nodes

# Find initializer shape
def get_initializer_shape(graph, initializer_name):
    for initializer in graph.initializer:
        if initializer.name == initializer_name:
            return [dim for dim in initializer.dims]
    return None

# Find node index
def find_node_index(graph_nodes, target_node):
    for index, node in enumerate(graph_nodes):
        if node == target_node:
            #print(f"node {node.name}: index {index}")
            return index
    return -1

# Find reshape nodes that follow other reshape nodes
def find_reshape_following_reshape(graph, reshape_node):
    for node in graph.node:
        # Check if the current node is a 'Reshape' operation and if the input of the reshape_node is in the output of any node
        if node.op_type == 'Reshape' and reshape_node.output[0] in node.input:
            # Return both the current reshape_node and the found following 'Reshape' node
            return (reshape_node, node)
    # Return the current reshape_node and None if no following 'Reshape' node is found
    return None

# Fuse matmul and add into gemm
def fuse_matmul_add(model: onnx.ModelProto):

    graph = model.graph
    
    for i, mm_node in enumerate(graph.node):
        print(f"Checking node {i}/{len(graph.node)}: {mm_node.op_type}")
        if mm_node.op_type != 'MatMul' or len(mm_node.output) != 1:
            continue

        print(f"Found a MatMul node: {mm_node.name}")

        add_node = find_following_add_node(graph, mm_node)
        if not add_node:
            print(f"No Add node follows the MatMul node: {mm_node.name}")
            continue

        print(f"Found a following Add node: {add_node.name}")

        # Create a Constant node that contains the goal shape
        shape = [-1, 768 ]
        shape_tensor = helper.make_tensor(
            name='shape_tensor',
            data_type=onnx.TensorProto.INT64,
            dims=[2],
            vals=shape
        )

        shape_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=[f'shape_{i}_output'],
            value=shape_tensor,
            name=f'shape_{i}'
        )

        print(f"Created a Constant node for shape: {shape_node.name}")

        # Create a Reshape node
        reshape_node = helper.make_node(
            'Reshape',
            inputs=[mm_node.input[0], f'shape_{i}_output'],
            outputs=[f'reshaped_output_{i}'],
            name=f'Reshape_{i}'
        )

        print(f"Created a Reshape node: {reshape_node.name}")

        # Insert the Constant and Reshape nodes into the graph
        graph.node.insert(i+1, shape_node)
        print(f"Inserted Constant node after node {i}: {shape_node.name}")
        graph.node.insert(i+2, reshape_node)
        print(f"Inserted Reshape node after node {i+1}: {reshape_node.name}")

        # Create a GEMM node to replace the MatMul and Add nodes
        gemm_inputs = [f'reshaped_output_{i}', mm_node.input[1],
                       add_node.input[1] if add_node.input[0] == mm_node.output[0] else add_node.input[0]]
        gemm_node = helper.make_node(
            'Gemm', # op type
            inputs=gemm_inputs,
            outputs=add_node.output,
            name=f'Gemm_{i}'
        )

        print(f"Created a GEMM node to replace MatMul and Add: {gemm_node.name}")

        # Replace the Add node with the GEMM node
        graph.node.insert(i+3, gemm_node)
        print(f"Inserted GEMM node after Reshape node: {gemm_node.name}")
        graph.node.remove(mm_node)
        print(f"Removed the original MatMul node: {mm_node.name}")
        graph.node.remove(add_node)
        print(f"Removed the original Add node: {add_node.name}")

# Replace any original reshapes that follow the gemm with reshape nodes that contain the correct shape dimensions
def replace_reshape_nodes(graph):
    for i, node in enumerate(graph.node):
        if node.op_type == 'Gemm':
            # Find all Reshape nodes that follow this Gemm node
            reshape_nodes = []
            next_node = find_following_reshape_node(graph, node)
            while next_node:
                reshape_nodes.append(next_node)
                next_node = find_following_reshape_node(graph, next_node)

            # If there are no Reshape nodes to replace, skip this Gemm node
            if not reshape_nodes:
                continue

            # Remove the Reshape nodes from the graph
            for reshape_node in reshape_nodes:
                # Find the associated Constant node
                constant_node = find_associated_constant_node(graph, reshape_node)
                if constant_node:
                    graph.node.remove(constant_node)

                graph.node.remove(reshape_node)

                # Create the shape tensor and shape node for 3D reshape
                shape_tensor_3d = helper.make_tensor("shape", onnx.TensorProto.INT64, [4], [1, 128, 12, 64])
                shape_node_3d = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[f'3d_shape_{i}_output'],
                    value=shape_tensor_3d,
                    name=f'shape_{i}_3d'
                )

                # Create the 3D Reshape node
                reshape_node_3d = helper.make_node(
                    'Reshape',
                    inputs=[node.output[0], f'3d_shape_{i}_output'],
                    outputs=[reshape_nodes[-1].output[0]],  # Use the output of the last Reshape node,
                    name=f'reshape_{i}_3d'
                )
                graph.node.insert(i+1, shape_node_3d)
                graph.node.insert(i+2, reshape_node_3d)

# Insert Reshapes following GEMM to reshape tensor back to 3D
def insert_reshape_nodes(graph):
    for i, node in enumerate(graph.node):
        if node.op_type == 'Gemm' and node.name != '/pooler/dense/Gemm':
            output_nodes = find_output_nodes(graph, node)
            if any(output_node.op_type != 'Reshape' for output_node in output_nodes):
                new_shape_tensor = helper.make_tensor("shape", onnx.TensorProto.INT64, [3], [1, 128, 768])
                new_shape_node = helper.make_node(
                    'Constant',
                    inputs=[],
                    outputs=[f'new_shape_{i}_output'],
                    value=new_shape_tensor,
                    name=f'new_shape_{i}'
                )
                new_reshape_node = helper.make_node(
                    'Reshape',
                    inputs=[node.output[0], f'new_shape_{i}_output'],
                    outputs=[f'new_reshaped_output_{i}'],
                    name=f'new_Reshape_{i}'
                )
                graph.node.insert(i+1, new_reshape_node)
                graph.node.insert(i+1, new_shape_node)
                for output_node in output_nodes:
                    output_node.input[0] = f'new_reshaped_output_{i}'

def modify_ffn_reshape_3d(graph):
    nodes_to_remove = []
    nodes_to_add = []

    for i, node in enumerate(graph.node):
        # Check in initializers
        for initializer in graph.initializer:
            #print(f"{initializer.name}: {initializer.dims}")
            if initializer.name in node.input and [dim for dim in initializer.dims] == [768, 3072] :
                reshape_nodes = []
                next_node = find_following_reshape_node(graph, node)
                while next_node:
                    reshape_nodes.append(next_node)
                    next_node = find_following_reshape_node(graph, next_node)

                if not reshape_nodes:
                    continue

                for reshape_node in reshape_nodes:
                    #print(f"node:{reshape_node.name}, input: {reshape_node.input[0]}, output: {reshape_node.output[0]}")
                    constant_node = find_associated_constant_node(graph, reshape_node)
                    if constant_node:
                        nodes_to_remove.append(constant_node)

                    nodes_to_remove.append(reshape_node)

                    shape_tensor_3d = helper.make_tensor("shape", onnx.TensorProto.INT64, [3], [1, 128, 3072])
                    shape_node_3d = helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[f'ffn_shape_{i}_output_3d'],
                        value=shape_tensor_3d,
                        name=f'shape_{i}_3d'
                    )
                    reshape_node_3d = helper.make_node(
                        'Reshape',
                        inputs=[node.output[0], f'ffn_shape_{i}_output_3d'],
                        outputs=[reshape_nodes[-1].output[0]],  # Use the output of the last Reshape node
                        name=f'ffn_reshape_{i}_3d'
                    )
                    nodes_to_add.append((i+1, shape_node_3d))
                    nodes_to_add.append((i+2, reshape_node_3d))
                    break

    # Remove nodes
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    # Add new nodes
    for index, node in nodes_to_add:
        graph.node.insert(index, node)

def modify_ffn_reshape_2d(graph):
    nodes_to_remove = []  # List to keep track of nodes to be removed
    nodes_to_add = []  # List to keep track of nodes to be added

    for i, node in enumerate(graph.node):
        # Check if node operation is 'Gemm'
        if node.op_type == 'Gemm':
            initializer_name = node.input[1]  # Get the name of the initializer for the 'Gemm' node
            initializer_shape = get_initializer_shape(graph, initializer_name)  # Get the shape of the initializer
            if initializer_shape == [3072, 768]:  # Check if shape matches specific criteria
                
                reshape_nodes = []
                prev_node = find_previous_reshape_node(graph, node)  # Find the previous reshape node connected to the current node
                while prev_node:  # Iterate through all connected reshape nodes
                    reshape_nodes.append(prev_node)
                    prev_node = find_previous_reshape_node(graph, prev_node)
                    
                if not reshape_nodes:  # If no reshape nodes are found, continue to the next node
                    continue

                for reshape_node in reshape_nodes:
                    constant_node = find_associated_constant_node(graph, reshape_node)  # Find constant node associated with reshape node
                    if constant_node:  # If a constant node is found, mark it for removal
                        nodes_to_remove.append(constant_node)

                    nodes_to_remove.append(reshape_node)  # Mark the reshape node for removal
                    
                    output_node = find_following_node(graph, reshape_node)  # Find the node that follows the reshape node

                    # Create a constant node for the new shape (3D tensor shape)
                    shape_tensor_3d = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [-1, 3072])
                    shape_node_3d = helper.make_node(
                        'Constant',
                        inputs=[],
                        outputs=[f'new_shape_{i}'],
                        value=shape_tensor_3d,
                        name=f'shape_{i}'
                    )
                    # Create a new reshape node to transform the tensor to the desired 3D shape
                    reshape_output_name = f'new_reshape_{i}'  # Unique output name for the new reshape node
                    reshape_node_3d = helper.make_node(
                        'Reshape',
                        inputs=[reshape_node.input[0], f'new_shape_{i}'],
                        outputs=[reshape_output_name],
                        name=f'new_reshape_{i}'
                    )

                    # Update the input of the node that follows the reshape node to accept the output of the new reshape node
                    for index, input_name in enumerate(output_node.input):
                        if input_name == reshape_node.output[0]:
                            output_node.input[index] = reshape_output_name
                            
                    nodes_to_add.append((i-2, reshape_node_3d))  # Schedule the new reshape node for insertion
                    nodes_to_add.append((i-5, shape_node_3d))  # Schedule the shape node for insertion
                    
                    break  # Exit the loop after handling the reshape nodes for the current 'Gemm' node
                        # Remove marked nodes from the graph
                
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
            #print(f'removed: {node.name}')

    # Insert new nodes into the graph at specified indices
    for index, node in nodes_to_add:
        graph.node.insert(index, node)
        #print(f'inserted: {node.name}')

def fuse_reshape_nodes(graph):
    for i, node in enumerate(graph.node):
        if node.op_type == 'Reshape':
            print(f"Processing {node.name}")
            reshape_nodes = [node]  # Start with the current node
            result = find_reshape_following_reshape(graph, node)
            while result:
                _, next_node = result  # Unpack the tuple
                if next_node in reshape_nodes:  # Prevent infinite loop by checking if next_node is already processed
                    print("Next node already processed, breaking loop to prevent infinite loop.")
                    break
                print(f"Found following reshape node: {next_node.name}")
                reshape_nodes.append(next_node)  # Store the next_node for processing
                result = find_reshape_following_reshape(graph, next_node)  # Use next_node for the next search

            if len(reshape_nodes) <= 1:  # Check if no additional 'Reshape' nodes were found
                continue

            # Remove the old 'Reshape' nodes and their associated 'Constant' nodes
            for reshape_node in reshape_nodes:
                print(f"Removing node: {reshape_node.name}")
                constant_node = find_associated_constant_node(graph, reshape_node)
                if constant_node:
                    graph.node.remove(constant_node)
                graph.node.remove(reshape_node)

            # After removal, create and insert the new 'Constant' and 'Reshape' nodes
            shape_tensor = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [128, 768])
            shape_node_name = f'fused_shape_{i}_output'  # Unique name for the shape node
            shape_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[shape_node_name],
                value=shape_tensor,
                name=f'fused_shape_{i}'
            )

            # Create the new 'Reshape' node
            new_reshape_node = helper.make_node(
                'Reshape',
                inputs=[reshape_nodes[0].input[0], shape_node_name],  # Input from first old 'Reshape', shape from new 'Constant'
                outputs=[reshape_nodes[-1].output[0]],  # Output to where the last old 'Reshape' connected
                name=f'fused_reshape_{i}'
            )

            # Insert the new nodes into the graph
            # Adjust the insertion point as needed based on your graph's structure
            graph.node.insert(i-1, new_reshape_node)
            graph.node.insert(i-1, shape_node)
            

model_path = 'bertsquad-12.onnx'
model = onnx.load(model_path)

fuse_matmul_add(model)

modify_ffn_reshape_2d(model.graph)

onnx.save(model, "fused-model.onnx")
print(f"Model after fusion saved. ")

fused_model = onnx.load("fused-model.onnx")
onnx.checker.check_model(fused_model)


