# Import necessary libraries
from flask import Flask, render_template, request
import scripts.TSF_GBT as TSF_GBT
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend to avoid threading issues

# Create a Flask web app
app = Flask(__name__)


# Function to run training procedure
def run_training():
    TSF_GBT.train()
    output = "Training completed."
    # return {'output': output}
    return {
        'output': output
    }


# Function to run inference code and generate output
def run_inference():
    fig, out_original, out = TSF_GBT.run_inference()
    # Example: Extract HTML table from the DataFrame
    html_table1 = out_original.to_html(index=True)
    html_table2 = out.to_html(index=True)

    # Save the plot to a BytesIO object
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    # Encode the image to base64 for rendering in HTML
    plot_data = base64.b64encode(img.getvalue()).decode()

    # Return the plot data and any other text/data
    return {
        'plot_data': plot_data,
        'html_table1': html_table1,
        'html_table2': html_table2
    }


# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    plot_data = None
    html_table1 = None
    html_table2 = None
    output = None

    button_type = request.form.get('button_type')

    if request.method == 'POST':
        # Run code only when the button is clicked (POST request)
        if button_type == 'button1':
            result = run_training()
            output = result['output']

        elif button_type == 'button2':
            result = run_inference()
            plot_data = result['plot_data']
            html_table1 = result['html_table1']
            html_table2 = result['html_table2']

    # Render the HTML template and pass plot_data and text_output
    return render_template('index.html', plot_data=plot_data, html_table1=html_table1,
                           html_table2=html_table2, output=output)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
