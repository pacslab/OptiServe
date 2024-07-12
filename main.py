from src.profiler.profiler import create_functions, profile_function, get_function_profiling_logs

# funcs = create_functions(application_dir='./experiments/applications/A1')

# profile_function(function=funcs[0], num_of_iterations=1)


print(get_function_profiling_logs(log_group_name='/aws/lambda/A1_F1'))
