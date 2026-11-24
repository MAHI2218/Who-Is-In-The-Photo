"""
Who Is in the Photo? - Offline Face Recognition Desktop Application
Entry Point.
"""

import sys
from PySide6.QtWidgets import QApplication
from app.ui_main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Who Is in the Photo?")
    app.setOrganizationName("Offline Face Recognition")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
