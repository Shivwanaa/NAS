import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp
import re
import streamlit as st
from io import BytesIO
stri="invalid"
# Function to validate the input transfer function
def validate_input(func_str):
    # Check for negative coefficients/terms
    if re.search(r'-\s*\d*\.?\d*\s*', func_str):  # Match negative numbers
        return False
    return True

# Function to extract coefficients from an expression using sympy
def parse_function(func_str):
    # Parse the function using sympy
    s = sp.symbols('s')
    if '/' not in func_str:
        num_expr=func_str
        den_expr='1'

    else: 
     num_expr, den_expr = func_str.split('/')
    # num_expr, den_expr = func_str.split('/')
    num_poly = sp.expand(sp.sympify(num_expr.strip()))

    den_poly = sp.expand(sp.sympify(den_expr.strip()))

    # Handle cases where the polynomial is a constant (degree 0)
    num_poly = sp.Poly(num_poly, s) if num_poly.has(s) else sp.Poly(num_poly, s, domain='QQ')
    den_poly = sp.Poly(den_poly, s) if den_poly.has(s) else sp.Poly(den_poly, s, domain='QQ')

    # Get the coefficients as lists
    num_coeffs = num_poly.all_coeffs()
    den_coeffs = den_poly.all_coeffs()

    # Convert to floats for signal.TransferFunction compatibility
    num_coeffs = [float(c) for c in num_coeffs]
    den_coeffs = [float(c) for c in den_coeffs]

    return num_coeffs, den_coeffs, num_poly, den_poly

# Function to perform polynomial division
def polynomial_division(numerator, denominator):
    """Perform polynomial division and return the quotient and remainder."""
    s = sp.symbols('s')
    quotient, remainder = sp.div(numerator, denominator)
    return quotient, remainder

# Function to decompose the transfer function into partial fractions
def partial_fraction_decomposition(function):
    """Decompose the function into partial fractions."""
    s = sp.symbols('s')
    numerator, denominator = function.as_numer_denom()

    # Check if the polynomials are constant and set degrees accordingly
    num_degree = numerator.as_poly(s).degree() if numerator.has(s) else 0
    den_degree = denominator.as_poly(s).degree() if denominator.has(s) else 0

    print("Degree of numerator:", num_degree)
    print("Degree of denominator:", den_degree)

    if num_degree > den_degree:
        quotient, remainder = polynomial_division(numerator, denominator)
        remainder_fraction = remainder / denominator
        decomposed_remainder = sp.apart(remainder_fraction)
        print("Decomposed Partial Fraction:")
        return quotient + decomposed_remainder
    else:
        return sp.apart(function)


# Function to check if poles and zeros are valid (on real or imaginary axes only)
def check_validity(zeros, poles):
    for z in zeros:
        if not (np.isclose(np.imag(z), 0) or np.isclose(np.real(z), 0)):
            return False
    for p in poles:
        if not (np.isclose(np.imag(p), 0) or np.isclose(np.real(p), 0)):
            return False
    return True

# Function to check if poles and zeros are alternating and identify the circuit type
def check_alternating_and_identify(poles, zeros):
    # Combine poles and zeros and sort them along the real and imaginary axes
    combined_points = [(z, 'zero') for z in zeros] + [(p, 'pole') for p in poles]
    combined_points.sort(key=lambda x: (np.real(x[0]), np.imag(x[0])))

    # Check for alternating pattern and identify the closest point to the origin
    last_type = None
    is_alternating = True
    closest_point = min(combined_points, key=lambda x: np.abs(x[0]))

    for point, point_type in combined_points:
        if last_type is None:
            last_type = point_type
        else:
            if last_type == point_type:
                is_alternating = False
                break
            last_type = point_type

    if is_alternating:
        # Determine axis type
        if np.all(np.isclose(np.imag([p[0] for p in combined_points]), 0)):
            axis_type = "real"
        elif np.all(np.isclose(np.real([p[0] for p in combined_points]), 0)):
            axis_type = "imaginary"
        else:
            axis_type = "mixed"

        # Identify circuit type based on the closest point to the origin
        if axis_type == "imaginary":
            print("LC circuit (alternating along the imaginary axis)")
            stri="LC"
            st.write("LC")
        elif axis_type == "real":
            if closest_point[1] == 'pole':
                print("RC circuit (alternating along the real axis, closest point is a pole)")
                stri="LC"
            elif closest_point[1] == 'zero':
                print("RL circuit (alternating along the real axis, closest point is a zero)")
                stri="LC"
        else:
            print("Mixed axis detected, cannot classify as a standard LC, RC, or RL circuit.")
            stri="Mixed axis detected, cannot classify as a standard LC, RC, or RL circuit."
    else:
        print(" ")
    # return stri


# Take user input for the transfer function
# func_str = input("Enter the transfer function in the format 'numerator/denominator' (e.g., '(s*3 + s) / ((s + 3)*3 * (s + 1))'): ")

# Validate input for negative terms
def get_circuit_type(func_str):
 if not validate_input(func_str):
    print("Invalid input: Transfer function should not contain negative terms.")
 else:
    # Parse the function to get numerator and denominator coefficients and expressions
    numerator, denominator, num_poly, den_poly = parse_function(func_str)

    # Create a transfer function system
    system = signal.TransferFunction(numerator, denominator)

    # Find the poles and zeros
    zeros = system.zeros
    poles = system.poles

    def format_values(values):
        return [complex(0, round(z.imag, 10)) if np.isclose(z.real, 0, atol=1e-10) else z for z in values]

    # Format the poles and zeros
    formatted_zeros = format_values(zeros)
    formatted_poles = format_values(poles)

    # Plotting the poles and zeros
    plt.figure()
    plt.scatter(np.real(formatted_zeros), np.imag(formatted_zeros), s=50, color='red', label='Zeroes', marker='o')
    plt.scatter(np.real(formatted_poles), np.imag(formatted_poles), s=50, color='blue', label='Poles', marker='x')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Poles and Zeroes Plot')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)  # Rewind the buffer to the beginning

    # Close the plot to release resources
    plt.close()
    st.image(buf)


    

    


    # Print partial fraction decomposition
    s = sp.symbols('s')
    transfer_function = num_poly/den_poly
    partial_fractions = partial_fraction_decomposition(transfer_function)
    print("\nPartial Fraction Decomposition:")
    print(partial_fractions)

    # Check the validity of poles and zeros
    if not check_validity(zeros, poles):
        print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
        stri="INVALID"
    else:
        # Check if poles and zeros are alternating and identify the circuit type
        stri=check_alternating_and_identify(poles, zeros)

    if not check_validity(formatted_zeros, formatted_poles):
        print("Invalid Case: Poles and zeros must be on the real or imaginary axes.")
    else:
        check_alternating_and_identify(formatted_poles, formatted_zeros)
    return(stri)
