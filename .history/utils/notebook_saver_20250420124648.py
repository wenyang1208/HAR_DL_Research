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
        filename = os.path.join(self.save_dir, 'notebook_code_snapshot.py')
        code_cells = get_ipython().user_ns['In']

        with open(filename, 'w', encoding='utf-8') as f:
            for cell in code_cells:
                if cell is not None:
                    f.write(cell + '\n\n')

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

    def save_plot(self, name='plot'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_dir, f'{name}_{timestamp}.png')
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")

        def save_plot(self, name='plot'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.save_dir, f'{name}_{timestamp}.png')
        plt.savefig(filename)
        print(f"Plot saved to: {filename}")

    def get_save_dir(self):
        return self.save_dir