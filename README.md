# Signature Recognition

A desktop application for signature recognition using Python, PyTorch, PyQT, and an LSTM model. This application allows users to train their own signature dataset and perform predictions.

## Features

- Train your own signature dataset.
- LSTM-based model for signature recognition.
- User-friendly GUI built with PyQT.

## Prerequisites

Ensure you have the following installed:

- Python 3.10 or higher
- PyTorch
- PyQT 6
- ConfigParser (for handling `config.ini`)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/signature-recognition.git
cd signature-recognition
```

2. Set up the `config.ini` file:

Create a `config.ini` file in the root directory with the following format:

```ini
[admin]
password=your_password_here
```
Replace `your_password_here` with your desired password.

## Usage

1. Run the main GUI application:

```bash
python main_gui.py
```

2. Train your signature dataset:
   - Follow the prompts in the GUI to add your signature data.
   - The application will automatically train the dataset using the LSTM model.

3. Predict signatures:
   - After training, use the GUI to predict new signatures.

## File Structure

```plaintext
.
├── main_gui.py         # Entry point for the GUI application
├── models/             # Directory containing the LSTM model files
├── dataset/            # Directory for storing signature datasets
├── predictions/        # Directory for storing signature predictions
├── config.ini          # Configuration file for user password
├── README.md           # Project documentation
```

## Contributing

Feel free to fork this repository, open issues, or submit pull requests.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

Special thanks to the open-source community for providing excellent tools and frameworks.
