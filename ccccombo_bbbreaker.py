import numpy as np

#set the boundry conditions of the equations & define some global constants
Bounds = np.array([[0,5], [0,5], [-5,5], [-5,0], [-5,0]])

real_value = 0.17
choices = 10
number_of_samples = 10

###############################################################################
###############################################################################
###############################################################################

#define a few functions to calculate and test inputs
def combo_finder(frac, iso):
    calc = np.zeros((5, choices))
    for z, i in enumerate(frac):
        calc[z] = frac[z] * iso[z]
    return calc

###############################################################################
###############################################################################
###############################################################################

#function to test the output from combo_finder
def solution_tester(input, solution):
    tst_output = np.zeros((len(input)))
    for idx, i in enumerate(input):
        test = sum(input[idx])
        if test == solution:
            print(idx, 'solution == input value')
            tst_output[idx] = test
        if test < solution:
            print(idx, 'too low')
            tst_output[idx] = test
        if test > solution:
            print(idx, 'too high')
            tst_output[idx] = test
    return  tst_output

###############################################################################
###############################################################################
###############################################################################

#function to use previous input if near correct value and try again
#pseudo machine learning
def solution_improvement():
    pass

###############################################################################
###############################################################################
###############################################################################

def choose_answers(boundry_array, iterate):

    #choose a set of values from the boundries to be tested
    if iterate == False:
        A_test = np.linspace(boundry_array[0,0], boundry_array[0,1], choices)
        B_test = np.linspace(boundry_array[1,0], boundry_array[1,1], choices)
        C_test = np.linspace(boundry_array[2,0], boundry_array[2,1], choices)
        D_test = np.linspace(boundry_array[3,0], boundry_array[3,1], choices)
        E_test = np.linspace(boundry_array[4,0], boundry_array[4,1], choices)
    if iterate == True:
        boundry_low = boundry_array * 0.8
        boundry_high = boundry_array * 1.2
        A_test = np.linspace(boundry_low[0], boundry_high[0], choices)
        B_test = np.linspace(boundry_low[1], boundry_high[1], choices)
        C_test = np.linspace(boundry_low[2], boundry_high[2], choices)
        D_test = np.linspace(boundry_low[3], boundry_high[3], choices)
        E_test = np.linspace(boundry_low[4], boundry_high[4], choices)

    #randomly (if you want) choose some values from this setup
    #pick some random numbers from linspace and assign them to an array
    rand_A = np.array([np.random.choice(choices) for i in range(number_of_samples)])
    rand_B = np.array([np.random.choice(choices) for i in range(number_of_samples)])
    rand_C = np.array([np.random.choice(choices) for i in range(number_of_samples)])
    rand_D = np.array([np.random.choice(choices) for i in range(number_of_samples)])
    rand_E = np.array([np.random.choice(choices) for i in range(number_of_samples)])

    #assign randomized values to arrays
    A_randomized = A_test[rand_A]
    B_randomized = B_test[rand_B]
    C_randomized = C_test[rand_C]
    D_randomized = D_test[rand_D]
    E_randomized = E_test[rand_E]

    return np.array([A_randomized, B_randomized, C_randomized, D_randomized, E_randomized])


#frac for my application has to ~ 1, not really needed in final code when I put in real data but helpful for testing
frac_list = []
example_frac = np.linspace(0, 1, 100)
rand_frac = np.random.choice(100)
alpha_value = example_frac[rand_frac]
frac_list.append(alpha_value)
left_over = 1 - alpha_value

for i in range(4):
    if left_over != 0:
        example_frac = np.linspace(0,left_over, 100)
        rand_frac = np.random.choice(100)
        alpha_value = example_frac[rand_frac]
        frac_list.append(alpha_value)
        left_over = 1 - sum(frac_list)

        #for the case that it randomly picks ~1, fill with zeros
    else:
        remainder = 5 - len(frac_list)
        for i in range(remainder):
            frac_list.append(0.0)
    #Finally randomize the order of the new list for frac
np.random.shuffle(frac_list)



###############################################################################
###############################################################################
###############################################################################

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

tester_array = choose_answers(Bounds, False)

final_values = combo_finder(frac_list, tester_array)
final_values = final_values.T

results = solution_tester(final_values, real_value)
new_boundry = np.zeros((5))

print('\n first trial \n')
#find results + or - 20% of the target value
for idx, i in enumerate(results):
    if (real_value * 1.5) > results[idx] and (real_value * 0.5) < results[idx]:
        print(idx, "{:.3f}".format(results[idx]), 'values:', tester_array.T[idx])
        new_boundry = tester_array.T[idx]
    else:
        print(idx, 'not within + or - 50%')

print('\n nth trial \n')

tester_array = choose_answers(new_boundry, True)

final_values = combo_finder(frac_list, tester_array)
final_values = final_values.T

results = solution_tester(final_values, real_value)

for idx, i in enumerate(results):
    if (real_value * 1.2) > results[idx] and (real_value * 0.8) < results[idx]:
        print(idx, "{:.3f}".format(results[idx]), 'values:', tester_array.T[idx])
        new_boundry = tester_array.T[idx]
    else:
        print(idx, 'not within + or - 20%')
