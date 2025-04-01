import tkinter as tk
import requests

def fetch_prediction():
    try:
        response = requests.get('http://127.0.0.1:4567/predict')
        data = response.json()
        result_var.set(f"11212: {data.get('error', 'Unknown error')}")
        print(data)
        if 'tomorrow_stiock_price' in data:
            result_var.set(f"Prediction: {data['tomorrow_stiock_price']}")
        else:
            result_var.set(f"Error: {data.get('error', 'Unknown error')}")
    except Exception as e:
        result_var.set(f"Error fetching data: {str(e)}")

# Set up the main window
root = tk.Tk()
root.title("Prediction Result")

# Create a label to display the results
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, wraplength=300)
result_label.pack(pady=20)

# Create a button to fetch predictions
fetch_button = tk.Button(root, text="Fetch Prediction", command=fetch_prediction)
fetch_button.pack(pady=10)

# Start the GUI loop
root.mainloop()