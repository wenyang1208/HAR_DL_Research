import matplotlib.pyplot as plt
import os
import contextlib
from datetime import datetime
from IPython import get_ipython

class NotebookSaver:
    def __init__(self, base_dir='notebook_versions'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(base_dir, f'{timestamp}_notebookversion')
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Save directory created: {self.save_dir}")

    def save_notebook_code(self):
        """Exports the current notebook's code cells to a Python file using nbconvert."""
        if not self.notebook_name:
            print("Error: Notebook name not set. Cannot export code.")
            return

        # Check if the notebook file exists (relative to where the script is run)
        # You might need to adjust the path logic depending on your project structure
        if not os.path.exists(self.notebook_name):
             print(f"Error: Notebook file '{self.notebook_name}' not found.")
             # Add more sophisticated path searching if needed
             return

        output_filename = os.path.join(self.save_dir, 'notebook_code_snapshot.py')
        output_filename_abs = os.path.abspath(output_filename) # Use absolute path for clarity

        # Construct the nbconvert command
        # Use '--output' to specify the exact output filename without '.ipynb' extension
        # Use '--output-dir' to ensure it goes to the correct directory
        command = [
            'jupyter', 'nbconvert',
            '--to', 'script',
            self.notebook_name, # The notebook file to convert
            '--output', os.path.splitext(os.path.basename(output_filename_abs))[0], # Base name without extension
            '--output-dir', os.path.dirname(output_filename_abs) # Directory part
        ]

        try:
            print(f"Running command: {' '.join(command)}")
            # Run the command
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Notebook code snapshot saved to: {output_filename_abs}")
            # print("nbconvert output:\n", result.stdout) # Uncomment for debugging
            # if result.stderr:
            #    print("nbconvert errors:\n", result.stderr) # Uncomment for debugging

        except FileNotFoundError:
             print("Error: 'jupyter' command not found. Make sure Jupyter and nbconvert are installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error during nbconvert execution:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Output:\n{e.output}")
            print(f"Stderr:\n{e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        print(f"Notebook code snapshot saved to: {filename}")

    def save_model_summary(self, model):
        save_path = os.path.join(self.save_dir, 'model_summary.txt')
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"Model summary saved to: {save_path}")

    def save_training_output(self, history, val_loss, val_acc):
        save_path = os.path.join(self.save_dir, 'training_output.txt')
        with open(save_path, 'w') as f:
            with contextlib.redirect_stdout(f):
                print("Training History:")
                for key in history.history:
                    print(f"{key}: {history.history[key]}")
                print(f"\nFinal Validation Loss: {val_loss}")
                print(f"Final Validation Accuracy: {val_acc}")
        print(f"Training output saved to: {save_path}")

    def save_plot(self, name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_dir, f'{name}_{timestamp}.png')
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")

    def get_save_dir(self):
        return self.save_dir