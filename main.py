from src.profiler.profiler import create_functions, profile_function

funcs = create_functions(application_dir='./experiments/applications/A1')

profile_function(function=funcs[0], num_of_iterations=1)