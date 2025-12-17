"""
NEVMY v15.1 - PyQt6 Interface
Нейросеть для распознавания рукописных цифр
Исправлены все баги интерфейса и переобучение
"""
import math
import random
import json
import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import List, Tuple, Optional

# PyQt6 импорты
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# ==================== КЛАСС ЛОГГЕРА ====================
class Logger:
    """Система логирования"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            self._initialized = True
    
    def setup_logging(self):
        """Настройка системы логирования"""
        # Определяем базовую директорию проекта
        if getattr(sys, 'frozen', False):
            # Если приложение собрано в exe
            base_dir = os.path.dirname(sys.executable)
        else:
            # Если запускаем из исходников
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Создаем папки если их нет
        log_dir = os.path.join(base_dir, "Logs")
        model_dir = os.path.join(base_dir, "Models")
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Формируем имя файла с датой
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"nevmy_{date_str}.log")
        
        # Настраиваем логгер
        self.logger = logging.getLogger("Nevmy")
        self.logger.setLevel(logging.DEBUG)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Файловый обработчик
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Очищаем старые обработчики и добавляем новые
        self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Сохраняем пути для использования в других классах
        self.base_dir = base_dir
        self.log_dir = log_dir
        self.model_dir = model_dir
    
    def get_base_dir(self):
        return self.base_dir
    
    def get_log_dir(self):
        return self.log_dir
    
    def get_model_dir(self):
        return self.model_dir
    
    def log(self, message: str, level: str = "INFO"):
        if level == "INFO":
            self.logger.info(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "DEBUG":
            self.logger.debug(message)

# ==================== УЛУЧШЕННАЯ НЕЙРОСЕТЬ ====================
class NevmyNetwork:
    """Улучшенная нейросеть для распознавания цифр"""
    def __init__(self, input_size=400, hidden_size=128, output_size=10, model_name="default"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.1
        self.model_name = model_name
        self.dropout_rate = 0.2  # Регуляризация Dropout
        self.l2_lambda = 0.001   # L2 регуляризация
        
        # Инициализация весов методом Xavier
        limit_ih = math.sqrt(6 / (input_size + hidden_size))
        limit_ho = math.sqrt(6 / (hidden_size + output_size))
        
        self.w1 = [[random.uniform(-limit_ih, limit_ih) for _ in range(hidden_size)] 
                  for _ in range(input_size)]
        self.w2 = [[random.uniform(-limit_ho, limit_ho) for _ in range(output_size)] 
                  for _ in range(hidden_size)]
        
        self.b1 = [0.0] * hidden_size
        self.b2 = [0.0] * output_size
        
        # Для отслеживания лучшей модели
        self.best_weights = None
        self.best_accuracy = 0.0
    
    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return 1.0 if x > 0 else 0.01
    
    @staticmethod
    def sigmoid(x):
        if x > 100: return 1.0
        if x < -100: return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    
    @staticmethod
    def softmax(values):
        max_val = max(values)
        exp_vals = [math.exp(v - max_val) for v in values]
        total = sum(exp_vals)
        if total == 0:
            return [1.0/len(values)] * len(values)
        return [v/total for v in exp_vals]
    
    def predict(self, inputs: List[float], training=False) -> List[float]:
        if len(inputs) != self.input_size:
            inputs = self._normalize_inputs(inputs)
        
        # Прямое распространение с ReLU
        hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            total = self.b1[j]
            for i in range(self.input_size):
                if i < len(inputs):
                    total += inputs[i] * self.w1[i][j]
            hidden[j] = self.relu(total)
        
        outputs = [0.0] * self.output_size
        for k in range(self.output_size):
            total = self.b2[k]
            for j in range(self.hidden_size):
                total += hidden[j] * self.w2[j][k]
            outputs[k] = total
        
        return self.softmax(outputs)
    
    def train_one(self, inputs: List[float], target: int) -> float:
        if len(inputs) != self.input_size:
            inputs = self._normalize_inputs(inputs)
        
        # Прямое распространение
        hidden = [0.0] * self.hidden_size
        hidden_raw = [0.0] * self.hidden_size
        
        for j in range(self.hidden_size):
            total = self.b1[j]
            for i in range(self.input_size):
                if i < len(inputs):
                    total += inputs[i] * self.w1[i][j]
            hidden_raw[j] = total
            hidden[j] = self.relu(total)
        
        # Dropout
        dropout_mask = [1.0] * self.hidden_size
        for j in range(self.hidden_size):
            if random.random() < self.dropout_rate:
                hidden[j] = 0.0
                dropout_mask[j] = 0.0
        
        outputs = [0.0] * self.output_size
        for k in range(self.output_size):
            total = self.b2[k]
            for j in range(self.hidden_size):
                total += hidden[j] * self.w2[j][k]
            outputs[k] = total
        
        probs = self.softmax(outputs)
        
        # Обратное распространение
        target_vec = [0.0] * self.output_size
        target_vec[target] = 1.0
        
        output_errors = [probs[k] - target_vec[k] for k in range(self.output_size)]
        
        # Добавляем L2 регуляризацию
        output_errors_l2 = [0.0] * self.output_size
        for k in range(self.output_size):
            output_errors_l2[k] = output_errors[k] + self.l2_lambda * self.b2[k]
        
        hidden_errors = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            error = 0.0
            for k in range(self.output_size):
                error += output_errors_l2[k] * self.w2[j][k]
            hidden_errors[j] = error * self.relu_derivative(hidden_raw[j]) * dropout_mask[j]
        
        hidden_errors_l2 = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            hidden_errors_l2[j] = hidden_errors[j] + self.l2_lambda * self.b1[j]
        
        # Обновление весов
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                grad = output_errors_l2[k] * hidden[j] + self.l2_lambda * self.w2[j][k]
                self.w2[j][k] -= self.learning_rate * grad
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                if i < len(inputs):
                    grad = hidden_errors_l2[j] * inputs[i] + self.l2_lambda * self.w1[i][j]
                    self.w1[i][j] -= self.learning_rate * grad
        
        for k in range(self.output_size):
            self.b2[k] -= self.learning_rate * output_errors_l2[k]
        
        for j in range(self.hidden_size):
            self.b1[j] -= self.learning_rate * hidden_errors_l2[j]
        
        return sum(abs(e) for e in output_errors) / self.output_size
    
    def save_best_model(self, accuracy: float):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_weights = {
                'w1': [row[:] for row in self.w1],
                'w2': [row[:] for row in self.w2],
                'b1': self.b1[:],
                'b2': self.b2[:]
            }
    
    def restore_best_model(self):
        if self.best_weights:
            self.w1 = [row[:] for row in self.best_weights['w1']]
            self.w2 = [row[:] for row in self.best_weights['w2']]
            self.b1 = self.best_weights['b1'][:]
            self.b2 = self.best_weights['b2'][:]
    
    def _normalize_inputs(self, inputs: List[float]) -> List[float]:
        if len(inputs) > self.input_size:
            return inputs[:self.input_size]
        elif len(inputs) < self.input_size:
            return inputs + [0.0] * (self.input_size - len(inputs))
        return inputs
    
    def save(self, filename: Optional[str] = None) -> str:
        logger = Logger()
        model_dir = logger.get_model_dir()
        
        if filename is None:
            filename = os.path.join(model_dir, f"{self.model_name}.json")
        
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'l2_lambda': self.l2_lambda,
            'w1': self.w1,
            'w2': self.w2,
            'b1': self.b1,
            'b2': self.b2,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)
        
        return f"✅ Модель сохранена: {filename}"
    
    def load(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.model_name = model_data.get('model_name', 'unnamed')
        self.learning_rate = model_data.get('learning_rate', 0.1)
        self.dropout_rate = model_data.get('dropout_rate', 0.2)
        self.l2_lambda = model_data.get('l2_lambda', 0.001)
        self.w1 = model_data['w1']
        self.w2 = model_data['w2']
        self.b1 = model_data['b1']
        self.b2 = model_data['b2']
        
        return f"✅ Модель загружена: {filename}"
    
    @staticmethod
    def get_available_models() -> List[str]:
        models = []
        try:
            logger = Logger()
            model_dir = logger.get_model_dir()
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                return models
                
            for file in os.listdir(model_dir):
                if file.endswith(".json"):
                    models.append(file[:-5])
        except Exception as e:
            print(f"Ошибка чтения моделей: {e}")
        return sorted(models)

# ==================== ГЛАВНОЕ ОКНО (ИСПРАВЛЕННОЕ) ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_model_name = None
        
        # Инициализация интерфейса БЕЗ show()
        self.init_ui()
        self.load_models_list()
        Logger().log("Приложение NEVMY запущено", "INFO")
        
        # Показываем окно только после полной инициализации
        QTimer.singleShot(100, self.finalize_ui)
    
    def finalize_ui(self):
        """Финальная настройка UI после загрузки"""
        self.show()
        self.setFocus()
    
    def init_ui(self):
        self.setWindowTitle("NEVMY v15.1 - Распознавание рукописных цифр")
        
        screen = QApplication.primaryScreen().geometry()
        width = min(1200, screen.width() - 100)
        height = min(800, screen.height() - 100)
        
        self.setGeometry(100, 100, width, height)
        
        # Центральный виджет
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        central_widget.setStyleSheet("""
            QWidget#centralWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2c3e50, stop:1 #1a252f);
            }
            QLabel {
                color: white;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Основной layout
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Заголовок
        title_label = QLabel("🧠 NEVMY v15.1")
        title_label.setObjectName("titleLabel")
        title_label.setStyleSheet("""
            QLabel#titleLabel {
                font-size: 48px;
                font-weight: bold;
                color: #3498db;
                padding: 20px;
                background-color: rgba(44, 62, 80, 0.8);
                border-radius: 25px;
                border: 4px solid #3498db;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle_label = QLabel("Нейросеть для распознавания рукописных цифр")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #ecf0f1;
                padding: 15px;
                font-weight: 500;
            }
        """)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(30)
        
        # Группа выбора модели
        model_group = QGroupBox("Управление моделями")
        model_group.setObjectName("modelGroup")
        model_group.setStyleSheet("""
            QGroupBox#modelGroup {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #3498db;
                border-radius: 20px;
                padding-top: 25px;
                margin-top: 15px;
                background-color: rgba(52, 73, 94, 0.4);
            }
            QGroupBox#modelGroup::title {
                subcontrol-origin: margin;
                left: 25px;
                padding: 0 20px 0 20px;
                background-color: #3498db;
                border-radius: 15px;
                color: white;
                font-size: 16px;
            }
        """)
        
        model_layout = QVBoxLayout()
        model_layout.setSpacing(15)
        
        # Строка выбора модели
        combo_layout = QHBoxLayout()
        combo_label = QLabel("Выберите модель:")
        combo_label.setStyleSheet("""
            QLabel {
                color: #ecf0f1;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        combo_layout.addWidget(combo_label)
        
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelCombo")
        self.model_combo.setMinimumHeight(45)
        self.model_combo.setStyleSheet("""
            QComboBox#modelCombo {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 12px;
                padding: 10px 15px;
                font-size: 15px;
                min-width: 300px;
            }
            QComboBox#modelCombo:hover {
                border: 2px solid #2980b9;
            }
            QComboBox#modelCombo::drop-down {
                border: none;
                width: 40px;
            }
            QComboBox#modelCombo::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid #ecf0f1;
                margin-right: 15px;
            }
            QComboBox#modelCombo QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #3498db;
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        
        self.refresh_btn = QPushButton("🔄 Обновить")
        self.refresh_btn.setObjectName("refreshBtn")
        self.refresh_btn.setFixedSize(120, 45)
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.setStyleSheet("""
            QPushButton#refreshBtn {
                background-color: #16a085;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#refreshBtn:hover {
                background-color: #138d75;
                border: 2px solid #1abc9c;
            }
            QPushButton#refreshBtn:pressed {
                background-color: #0e6655;
                padding-top: 2px;
                padding-left: 2px;
            }
        """)
        self.refresh_btn.clicked.connect(self.load_models_list)
        
        combo_layout.addWidget(self.model_combo, 1)
        combo_layout.addWidget(self.refresh_btn)
        combo_layout.addStretch()
        
        model_layout.addLayout(combo_layout)
        
        # Кнопки управления
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.load_btn = QPushButton("📂 Загрузить модель")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.setMinimumHeight(50)
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.clicked.connect(self.load_selected_model)
        
        self.delete_btn = QPushButton("🗑 Удалить модель")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.setMinimumHeight(50)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.clicked.connect(self.delete_model)
        
        for btn in [self.load_btn, self.delete_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #3498db;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 12px 25px;
                    font-size: 15px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #2980b9;
                    border: 2px solid #7fb3d5;
                }}
                QPushButton:pressed {{
                    background-color: #21618c;
                    padding-top: 14px;
                    padding-left: 27px;
                }}
            """)
        
        self.delete_btn.setStyleSheet("""
            QPushButton#deleteBtn {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 25px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton#deleteBtn:hover {
                background-color: #c0392b;
                border: 2px solid #ec7063;
            }
            QPushButton#deleteBtn:pressed {
                background-color: #922b21;
                padding-top: 14px;
                padding-left: 27px;
            }
        """)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.delete_btn)
        button_layout.addStretch()
        
        model_layout.addLayout(button_layout)
        
        # Статус модели
        self.model_status = QLabel("Модель не загружена")
        self.model_status.setObjectName("modelStatus")
        self.model_status.setMinimumHeight(60)
        self.model_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_status("not_loaded")
        model_layout.addWidget(self.model_status)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        layout.addSpacing(30)
        
        # Кнопки меню - ИСПРАВЛЕННЫЕ
        menu_layout = QGridLayout()
        menu_layout.setSpacing(20)
        menu_layout.setContentsMargins(30, 30, 30, 30)
        
        buttons = [
            ("🎨 Рисовать цифру", self.open_drawing, "#9b59b6"),
            ("🧠 Обучить модель", self.open_training, "#2ecc71"),
            ("📊 Тестировать модель", self.open_testing, "#e67e22"),
            ("📈 Показать логи", self.show_logs, "#3498db"),
            ("⚙ Настройки", self.show_settings, "#95a5a6"),
            ("❌ Выход", self.close, "#e74c3c")
        ]
        
        for i, (text, slot, color) in enumerate(buttons):
            btn = StableButton(text, color)
            btn.setMinimumHeight(90)
            btn.clicked.connect(slot)
            menu_layout.addWidget(btn, i // 2, i % 2)
        
        layout.addLayout(menu_layout)
        layout.addStretch()
        
        # Информационная панель
        info_label = QLabel("Архитектура: 400-128-10 нейронов | Сохранение: Models/ | Логи: Logs/")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #bdc3c7;
                padding: 15px;
                background-color: rgba(52, 73, 94, 0.4);
                border-radius: 12px;
                border: 1px solid #34495e;
            }
        """)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
        
        # Статус бар
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1a252f;
                color: #ecf0f1;
                font-size: 13px;
                border-top: 1px solid #34495e;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готов к работе")
        
        # Устанавливаем стили для всего приложения
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:disabled {
                background-color: #7f8c8d !important;
                color: #bdc3c7 !important;
                border: 2px solid #95a5a6 !important;
            }
        """)
        
        # Устанавливаем минимальный размер
        self.setMinimumSize(900, 700)
    
    def update_status(self, status_type: str, message: str = ""):
        if status_type == "not_loaded":
            self.model_status.setText("Модель не загружена")
            self.model_status.setStyleSheet("""
                QLabel {
                    padding: 15px;
                    font-size: 15px;
                    color: #e74c3c;
                    background-color: rgba(231, 76, 60, 0.15);
                    border-radius: 12px;
                    border: 2px solid #e74c3c;
                    font-weight: bold;
                }
            """)
        elif status_type == "loaded":
            self.model_status.setText(f"✅ Модель '{message}' загружена")
            self.model_status.setStyleSheet("""
                QLabel {
                    padding: 15px;
                    font-size: 15px;
                    color: #27ae60;
                    background-color: rgba(39, 174, 96, 0.15);
                    border-radius: 12px;
                    border: 2px solid #27ae60;
                    font-weight: bold;
                }
            """)
        elif status_type == "error":
            self.model_status.setText(f"❌ {message[:100]}")
            self.model_status.setStyleSheet("""
                QLabel {
                    padding: 15px;
                    font-size: 15px;
                    color: #e74c3c;
                    background-color: rgba(231, 76, 60, 0.15);
                    border-radius: 12px;
                    border: 2px solid #e74c3c;
                    font-weight: bold;
                }
            """)
    
    def load_models_list(self):
        models = NevmyNetwork.get_available_models()
        self.model_combo.clear()
        
        if models:
            self.model_combo.addItems(models)
            self.status_bar.showMessage(f"Найдено моделей: {len(models)}", 3000)
        else:
            self.model_combo.addItem("Модели не найдены")
            self.status_bar.showMessage("Модели не найдены. Создайте новую модель.", 3000)
    
    def load_selected_model(self):
        model_name = self.model_combo.currentText()
        
        if model_name == "Модели не найдены":
            QMessageBox.warning(self, "Внимание", "Нет доступных моделей!")
            return
        
        try:
            logger = Logger()
            model_dir = logger.get_model_dir()
            model_path = os.path.join(model_dir, f"{model_name}.json")
            
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Ошибка", f"Файл модели не найден: {model_path}")
                return
            
            self.model = NevmyNetwork()
            self.model.load(model_path)
            self.current_model_name = model_name
            
            self.update_status("loaded", model_name)
            Logger().log(f"Модель '{model_name}' загружена", "INFO")
            self.status_bar.showMessage(f"Модель '{model_name}' загружена", 3000)
            
        except Exception as e:
            self.update_status("error", f"Ошибка загрузки: {str(e)}")
            Logger().log(f"Ошибка загрузки модели: {e}", "ERROR")
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{str(e)}")
    
    def delete_model(self):
        model_name = self.model_combo.currentText()
        
        if model_name == "Модели не найдены":
            return
        
        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Вы уверены, что хотите удалить модель '{model_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                logger = Logger()
                model_dir = logger.get_model_dir()
                model_path = os.path.join(model_dir, f"{model_name}.json")
                
                if os.path.exists(model_path):
                    os.remove(model_path)
                    self.load_models_list()
                    self.update_status("not_loaded")
                    
                    if self.current_model_name == model_name:
                        self.model = None
                        self.current_model_name = None
                    
                    Logger().log(f"Модель '{model_name}' удалена", "INFO")
                    self.status_bar.showMessage(f"Модель '{model_name}' удалена", 3000)
                else:
                    QMessageBox.warning(self, "Ошибка", f"Файл модели не найден")
                    
            except Exception as e:
                Logger().log(f"Ошибка удаления модели: {e}", "ERROR")
                QMessageBox.critical(self, "Ошибка", f"Не удалось удалить модель:\n{str(e)}")
    
    def open_drawing(self):
        if self.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return
        
        self.drawing_window = DrawingWindow(self.model, self)
        self.hide()
        QTimer.singleShot(100, self.drawing_window.show)
    
    def open_training(self):
        self.training_window = TrainingWindow(self)
        self.hide()
        QTimer.singleShot(100, self.training_window.show)
    
    def open_testing(self):
        if self.model is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите модель!")
            return
        
        self.testing_window = TestingWindow(self.model, self)
        self.hide()
        QTimer.singleShot(100, self.testing_window.show)
    
    def show_logs(self):
        dialog = LogDialog(self)
        dialog.exec()
    
    def show_settings(self):
        QMessageBox.information(self, "Настройки", 
                              "Настройки будут доступны в следующей версии.\n\n"
                              "Текущие возможности:\n"
                              "• Темная тема\n"
                              "• Анимации интерфейса\n"
                              "• Сохранение моделей в формате JSON")
    
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Выход",
            "Вы уверены, что хотите выйти?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            Logger().log("Приложение закрыто", "INFO")
            event.accept()
        else:
            event.ignore()

# ==================== СТАБИЛЬНАЯ КНОПКА (БЕЗ ТРЯСКИ) ====================
class StableButton(QPushButton):
    """Кнопка без анимации тряски"""
    def __init__(self, text="", color="#3498db"):
        super().__init__(text)
        self.base_color = color
        self.hover_color = self._darken_color(color, 20)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Устанавливаем фиксированный стиль
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.base_color};
                color: white;
                border: none;
                border-radius: 15px;
                padding: 20px;
                font-size: 16px;
                font-weight: bold;
                min-height: 90px;
            }}
            QPushButton:hover {{
                background-color: {self.hover_color};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 40)};
                padding-top: 22px;
                padding-left: 22px;
            }}
        """)
    
    @staticmethod
    def _darken_color(color: str, amount=20) -> str:
        if color.startswith("#"):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            r = max(0, r - amount)
            g = max(0, g - amount)
            b = max(0, b - amount)
            return f"#{r:02x}{g:02x}{b:02x}"
        return color
    
    # Отключаем все события анимации
    def enterEvent(self, event):
        pass
    
    def leaveEvent(self, event):
        pass

# ==================== ОКНО РИСОВАНИЯ (ИСПРАВЛЕННОЕ) ====================
class DrawingWindow(QMainWindow):
    def __init__(self, model, main_window):
        super().__init__()
        self.model = model
        self.main_window = main_window
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("NEVMY - Рисование цифр")
        self.setGeometry(150, 150, 1100, 750)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2c3e50, stop:1 #1a252f);
            }
        """)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(25, 25, 25, 25)
        
        left_panel = QGroupBox("Область рисования")
        left_panel.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #9b59b6;
                border-radius: 15px;
                padding-top: 20px;
                background-color: rgba(52, 73, 94, 0.3);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 15px 0 15px;
                background-color: #9b59b6;
                border-radius: 10px;
                color: white;
            }
        """)
        
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        
        self.canvas = DrawingCanvas()
        self.canvas.setMinimumSize(550, 550)
        self.canvas.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 4px solid #7f8c8d;
                border-radius: 15px;
            }
        """)
        left_layout.addWidget(self.canvas)
        
        brush_panel = QGroupBox("Настройки кисти")
        brush_panel.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #95a5a6;
                border-radius: 10px;
                padding-top: 15px;
                background-color: rgba(52, 73, 94, 0.2);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                background-color: #95a5a6;
                border-radius: 5px;
                color: white;
            }
        """)
        
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Размер:"))
        
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(10, 50)
        self.size_slider.setValue(25)
        self.size_slider.setMinimumWidth(150)
        self.size_slider.valueChanged.connect(self.change_pen_size)
        
        self.size_label = QLabel("25 px")
        self.size_label.setMinimumWidth(50)
        
        brush_layout.addWidget(self.size_slider)
        brush_layout.addWidget(self.size_label)
        brush_layout.addStretch()
        brush_panel.setLayout(brush_layout)
        left_layout.addWidget(brush_panel)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        self.recognize_btn = StableButton("🔍 Распознать", "#27ae60")
        self.recognize_btn.setMinimumHeight(50)
        self.recognize_btn.clicked.connect(self.recognize_digit)
        
        self.clear_btn = StableButton("🧹 Очистить", "#e74c3c")
        self.clear_btn.setMinimumHeight(50)
        self.clear_btn.clicked.connect(self.clear_canvas)
        
        self.back_btn = StableButton("← Назад", "#3498db")
        self.back_btn.setMinimumHeight(50)
        self.back_btn.clicked.connect(self.go_back)
        
        button_layout.addWidget(self.recognize_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.back_btn)
        left_layout.addLayout(button_layout)
        
        left_panel.setLayout(left_layout)
        
        right_panel = QGroupBox("Результаты распознавания")
        right_panel.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #3498db;
                border-radius: 15px;
                padding-top: 20px;
                background-color: rgba(52, 73, 94, 0.3);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 15px 0 15px;
                background-color: #3498db;
                border-radius: 10px;
                color: white;
            }
        """)
        
        right_layout = QVBoxLayout()
        right_layout.setSpacing(20)
        
        self.result_label = QLabel("Нарисуйте цифру")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 96px;
                font-weight: bold;
                color: #3498db;
                qproperty-alignment: AlignCenter;
                padding: 40px;
                background-color: rgba(52, 73, 94, 0.3);
                border-radius: 20px;
                border: 3px solid #3498db;
                min-height: 200px;
            }
        """)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #bdc3c7;
                qproperty-alignment: AlignCenter;
                padding: 15px;
                font-weight: bold;
            }
        """)
        
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("Вероятности для каждой цифры:"))
        
        self.progress_bars = []
        self.percent_labels = []
        
        for i in range(10):
            bar_widget = QWidget()
            bar_layout = QHBoxLayout(bar_widget)
            bar_layout.setContentsMargins(0, 5, 0, 5)
            
            digit_label = QLabel(f"{i}:")
            digit_label.setFixedWidth(40)
            digit_label.setStyleSheet("color: #ecf0f1; font-weight: bold; font-size: 16px;")
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(False)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #34495e;
                    border-radius: 5px;
                    background-color: #2c3e50;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 4px;
                }
            """)
            
            percent_label = QLabel("0%")
            percent_label.setFixedWidth(60)
            percent_label.setStyleSheet("color: #bdc3c7; font-weight: bold;")
            
            bar_layout.addWidget(digit_label)
            bar_layout.addWidget(progress_bar)
            bar_layout.addWidget(percent_label)
            
            right_layout.addWidget(bar_widget)
            
            self.progress_bars.append(progress_bar)
            self.percent_labels.append(percent_label)
        
        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.setMinimumSize(1000, 650)
    
    def change_pen_size(self, value):
        self.canvas.pen_size = value
        self.size_label.setText(f"{value} px")
    
    def recognize_digit(self):
        if self.canvas.image.isNull():
            QMessageBox.warning(self, "Внимание", "Сначала нарисуйте цифру!")
            return
        
        pixels = self.canvas.get_pixels_for_neural_network()
        
        try:
            predictions = self.model.predict(pixels)
            predicted_digit = predictions.index(max(predictions))
            confidence = max(predictions) * 100
            
            if confidence > 85:
                color = "#27ae60"
                conf_text = f"✅ Высокая уверенность: {confidence:.1f}%"
            elif confidence > 65:
                color = "#f39c12"
                conf_text = f"⚠ Средняя уверенность: {confidence:.1f}%"
            else:
                color = "#e74c3c"
                conf_text = f"❌ Низкая уверенность: {confidence:.1f}%"
            
            self.result_label.setText(str(predicted_digit))
            self.result_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 96px;
                    font-weight: bold;
                    color: {color};
                    qproperty-alignment: AlignCenter;
                    padding: 40px;
                    background-color: rgba(52, 73, 94, 0.3);
                    border-radius: 20px;
                    border: 3px solid {color};
                    min-height: 200px;
                }}
            """)
            self.confidence_label.setText(conf_text)
            
            for i in range(10):
                percent = predictions[i] * 100
                self.progress_bars[i].setValue(int(percent))
                self.percent_labels[i].setText(f"{percent:.1f}%")
            
            Logger().log(f"Распознана цифра: {predicted_digit} (уверенность: {confidence:.1f}%)", "INFO")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось распознать цифру:\n{str(e)}")
            Logger().log(f"Ошибка распознавания: {e}", "ERROR")
    
    def clear_canvas(self):
        self.canvas.clear()
        self.result_label.setText("Нарисуйте цифру")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 96px;
                font-weight: bold;
                color: #3498db;
                qproperty-alignment: AlignCenter;
                padding: 40px;
                background-color: rgba(52, 73, 94, 0.3);
                border-radius: 20px;
                border: 3px solid #3498db;
                min-height: 200px;
            }
        """)
        self.confidence_label.setText("")
        
        for i in range(10):
            self.progress_bars[i].setValue(0)
            self.percent_labels[i].setText("0%")
    
    def go_back(self):
        self.main_window.show()
        self.close()

# ==================== ХОЛСТ ДЛЯ РИСОВАНИЯ ====================
class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.drawing = False
        self.pen_size = 25
        self.last_point = QPoint()
        
        self.image_size = 500
        self.network_size = 20
        
        self.image = QImage(self.image_size, self.image_size, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.update_pixmap()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.draw_point(event.pos())
    
    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            self.draw_line(self.last_point, event.pos())
            self.last_point = event.pos()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
    
    def draw_point(self, point):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.GlobalColor.black, self.pen_size, 
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, 
                           Qt.PenJoinStyle.RoundJoin))
        painter.drawPoint(point)
        painter.end()
        self.update_pixmap()
    
    def draw_line(self, from_point, to_point):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.GlobalColor.black, self.pen_size,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                           Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(from_point, to_point)
        painter.end()
        self.update_pixmap()
    
    def update_pixmap(self):
        pixmap = QPixmap.fromImage(self.image)
        self.setPixmap(pixmap)
    
    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update_pixmap()
    
    def get_pixels_for_neural_network(self):
        scaled = self.image.scaled(self.network_size, self.network_size,
                                  Qt.AspectRatioMode.IgnoreAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        
        pixels = []
        
        for y in range(self.network_size):
            for x in range(self.network_size):
                color = scaled.pixelColor(x, y)
                gray = (color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114) / 255.0
                gray = 1.0 - gray
                pixels.append(gray)
        
        if len(pixels) != 400:
            if len(pixels) > 400:
                pixels = pixels[:400]
            else:
                pixels = pixels + [0.0] * (400 - len(pixels))
        
        for i in range(len(pixels)):
            if pixels[i] < 0.1:
                pixels[i] = random.uniform(0, 0.05)
            else:
                pixels[i] = min(1.0, pixels[i] * 1.2)
                pixels[i] += random.uniform(-0.05, 0.05)
                pixels[i] = max(0.0, min(1.0, pixels[i]))
        
        return pixels

# ==================== ОКНО ОБУЧЕНИЯ (ИСПРАВЛЕННОЕ) ====================
class TrainingWindow(QMainWindow):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = None
        self.training_thread = None
        self.stop_training = False
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("NEVMY - Обучение модели")
        self.setGeometry(150, 150, 1000, 750)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2c3e50, stop:1 #1a252f);
            }
        """)
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        title_label = QLabel("🧠 Обучение новой модели")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #2ecc71;
                padding: 20px;
                background-color: rgba(46, 204, 113, 0.15);
                border-radius: 20px;
                border: 3px solid #2ecc71;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        params_group = QGroupBox("Параметры обучения")
        params_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #ecf0f1;
                border: 3px solid #3498db;
                border-radius: 15px;
                padding-top: 20px;
                background-color: rgba(52, 73, 94, 0.3);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 15px 0 15px;
                background-color: #3498db;
                border-radius: 10px;
                color: white;
            }
        """)
        
        params_layout = QGridLayout()
        params_layout.setSpacing(15)
        params_layout.setContentsMargins(25, 35, 25, 25)
        
        params_layout.addWidget(QLabel("Название модели:"), 0, 0)
        self.model_name_input = QLineEdit(f"model_{datetime.now().strftime('%Y%m%d_%H%M')}")
        self.model_name_input.setMinimumHeight(40)
        self.model_name_input.setStyleSheet("""
            QLineEdit {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.model_name_input, 0, 1)
        
        params_layout.addWidget(QLabel("Эпох обучения:"), 1, 0)
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(50)
        self.epochs_input.setMinimumHeight(40)
        self.epochs_input.setStyleSheet("""
            QSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.epochs_input, 1, 1)
        
        params_layout.addWidget(QLabel("Примеров на цифру:"), 2, 0)
        self.samples_input = QSpinBox()
        self.samples_input.setRange(10, 1000)
        self.samples_input.setValue(200)
        self.samples_input.setMinimumHeight(40)
        self.samples_input.setStyleSheet("""
            QSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.samples_input, 2, 1)
        
        params_layout.addWidget(QLabel("Скрытых нейронов:"), 3, 0)
        self.hidden_input = QSpinBox()
        self.hidden_input.setRange(10, 1000)
        self.hidden_input.setValue(128)
        self.hidden_input.setMinimumHeight(40)
        self.hidden_input.setStyleSheet("""
            QSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.hidden_input, 3, 1)
        
        params_layout.addWidget(QLabel("Скорость обучения:"), 4, 0)
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.001, 1.0)
        self.lr_input.setValue(0.05)
        self.lr_input.setSingleStep(0.01)
        self.lr_input.setMinimumHeight(40)
        self.lr_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.lr_input, 4, 1)
        
        params_layout.addWidget(QLabel("Dropout регуляризация:"), 5, 0)
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.0, 0.5)
        self.dropout_input.setValue(0.2)
        self.dropout_input.setSingleStep(0.05)
        self.dropout_input.setMinimumHeight(40)
        self.dropout_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.dropout_input, 5, 1)
        
        params_layout.addWidget(QLabel("L2 регуляризация:"), 6, 0)
        self.l2_input = QDoubleSpinBox()
        self.l2_input.setRange(0.0, 0.01)
        self.l2_input.setValue(0.001)
        self.l2_input.setSingleStep(0.0005)
        self.l2_input.setDecimals(4)
        self.l2_input.setMinimumHeight(40)
        self.l2_input.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        params_layout.addWidget(self.l2_input, 6, 1)
        
        params_layout.addWidget(QLabel("Ранняя остановка:"), 7, 0)
        self.early_stop_check = QCheckBox("Включить")
        self.early_stop_check.setChecked(True)
        self.early_stop_check.setStyleSheet("""
            QCheckBox {
                color: #ecf0f1;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
        """)
        params_layout.addWidget(self.early_stop_check, 7, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Прогресс: %p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 3px solid #3498db;
                border-radius: 10px;
                background-color: #2c3e50;
                color: white;
                text-align: center;
                font-size: 14px;
                font-weight: bold;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #3498db, stop:1 #2980b9);
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a252f;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        layout.addWidget(self.log_text)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = StableButton("🚀 Начать обучение", "#27ae60")
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = StableButton("⏹ Остановить", "#e74c3c")
        self.stop_btn.clicked.connect(self.stop_training_func)
        self.stop_btn.setEnabled(False)
        
        self.back_btn = StableButton("← Назад", "#3498db")
        self.back_btn.clicked.connect(self.go_back)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.back_btn)
        layout.addLayout(button_layout)
        
        self.setMinimumSize(900, 650)
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        QApplication.processEvents()
    
    def start_training(self):
        model_name = self.model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Ошибка", "Введите название модели!")
            return
        
        self.stop_training = False
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.log_text.clear()
        self.log("=" * 70)
        self.log(" " * 10 + "НАЧАЛО ОБУЧЕНИЯ НОВОЙ МОДЕЛИ")
        self.log("=" * 70)
        
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(model_name,),
            daemon=True
        )
        self.training_thread.start()
    
    def train_model_thread(self, model_name: str):
        try:
            epochs = self.epochs_input.value()
            samples_per_digit = self.samples_input.value()
            hidden_size = self.hidden_input.value()
            learning_rate = self.lr_input.value()
            dropout_rate = self.dropout_input.value()
            l2_lambda = self.l2_input.value()
            use_early_stop = self.early_stop_check.isChecked()
            
            self.log(f"\n📊 Параметры обучения:")
            self.log(f"   • Название модели: {model_name}")
            self.log(f"   • Эпох: {epochs}")
            self.log(f"   • Примеров на цифру: {samples_per_digit}")
            self.log(f"   • Скрытых нейронов: {hidden_size}")
            self.log(f"   • Скорость обучения: {learning_rate}")
            self.log(f"   • Dropout: {dropout_rate}")
            self.log(f"   • L2 регуляризация: {l2_lambda}")
            self.log(f"   • Ранняя остановка: {'Да' if use_early_stop else 'Нет'}")
            
            self.model = NevmyNetwork(
                input_size=400,
                hidden_size=hidden_size,
                output_size=10,
                model_name=model_name
            )
            self.model.learning_rate = learning_rate
            self.model.dropout_rate = dropout_rate
            self.model.l2_lambda = l2_lambda
            
            self.log("\n🔄 Создание данных...")
            
            train_size = int(samples_per_digit * 0.7)
            val_size = int(samples_per_digit * 0.15)
            test_size = samples_per_digit - train_size - val_size
            
            train_data, train_labels = self.create_training_data(train_size)
            val_data, val_labels = self.create_test_data(val_size)
            test_data, test_labels = self.create_test_data(test_size)
            
            self.log(f"✅ Создано данных:")
            self.log(f"   • Обучающих: {len(train_data)} примеров")
            self.log(f"   • Валидационных: {len(val_data)} примеров")
            self.log(f"   • Тестовых: {len(test_data)} примеров")
            
            self.log("\n🧠 Начинаю обучение...")
            self.log("-" * 70)
            
            best_val_accuracy = 0
            best_epoch = 0
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(epochs):
                if self.stop_training:
                    self.log("\n⚠ Обучение остановлено пользователем")
                    break
                
                total_error = 0
                train_correct = 0
                
                indices = list(range(len(train_data)))
                random.shuffle(indices)
                
                batch_size = 32
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    
                    for idx in batch_indices:
                        error = self.model.train_one(train_data[idx], train_labels[idx])
                        total_error += error
                
                for idx in range(len(train_data)):
                    probs = self.model.predict(train_data[idx], training=False)
                    if probs.index(max(probs)) == train_labels[idx]:
                        train_correct += 1
                
                train_accuracy = train_correct / len(train_data) * 100
                
                val_correct = 0
                for i in range(len(val_data)):
                    probs = self.model.predict(val_data[i], training=False)
                    if probs.index(max(probs)) == val_labels[i]:
                        val_correct += 1
                
                val_accuracy = val_correct / len(val_data) * 100
                
                self.model.save_best_model(val_accuracy)
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_epoch = epoch + 1
                    patience_counter = 0
                    log_msg = f"Эпоха {epoch+1:3d}/{epochs} | "
                    log_msg += f"Ошибка: {total_error/len(train_data):.4f} | "
                    log_msg += f"Точность: обуч. {train_accuracy:5.1f}%, вал. {val_accuracy:5.1f}% 🏆"
                else:
                    patience_counter += 1
                    log_msg = f"Эпоха {epoch+1:3d}/{epochs} | "
                    log_msg += f"Ошибка: {total_error/len(train_data):.4f} | "
                    log_msg += f"Точность: обуч. {train_accuracy:5.1f}%, вал. {val_accuracy:5.1f}%"
                
                self.log(log_msg)
                
                if use_early_stop and patience_counter >= max_patience:
                    self.log(f"\n⚠ Ранняя остановка: нет улучшений {max_patience} эпох")
                    self.log(f"   Лучшая точность: {best_val_accuracy:.1f}% (эпоха {best_epoch})")
                    break
                
                if (epoch + 1) % 20 == 0:
                    old_lr = self.model.learning_rate
                    self.model.learning_rate *= 0.8
                    self.log(f"   Скорость обучения: {old_lr:.4f} → {self.model.learning_rate:.4f}")
                
                progress = int((epoch + 1) / epochs * 100)
                self.progress_bar.setValue(progress)
                
                time.sleep(0.01)
            
            self.log("-" * 70)
            
            self.model.restore_best_model()
            
            test_correct = 0
            for i in range(len(test_data)):
                probs = self.model.predict(test_data[i], training=False)
                if probs.index(max(probs)) == test_labels[i]:
                    test_correct += 1
            
            test_accuracy = test_correct / len(test_data) * 100
            
            self.log(f"\n📈 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
            self.log(f"   • Лучшая точность на валидации: {best_val_accuracy:.1f}% (эпоха {best_epoch})")
            self.log(f"   • Точность на тесте: {test_accuracy:.1f}%")
            
            if not self.stop_training:
                result = self.model.save()
                self.log(f"\n{result}")
                
                self.log("\n" + "=" * 70)
                self.log(" " * 10 + "✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
                self.log("=" * 70)
                
                self.main_window.model = self.model
                self.main_window.current_model_name = model_name
                self.main_window.load_models_list()
                
                Logger().log(f"Модель '{model_name}' обучена. "
                           f"Валидация: {best_val_accuracy:.1f}%, Тест: {test_accuracy:.1f}%", "INFO")
                
                QMetaObject.invokeMethod(self, "show_success_dialog", 
                                        Qt.ConnectionType.QueuedConnection,
                                        Q_ARG(float, test_accuracy))
            
        except Exception as e:
            self.log(f"\n❌ ОШИБКА ПРИ ОБУЧЕНИИ: {str(e)}")
            Logger().log(f"Ошибка обучения: {e}", "ERROR")
        finally:
            QMetaObject.invokeMethod(self, "enable_start_button", 
                                    Qt.ConnectionType.QueuedConnection)
    
    def create_training_data(self, samples_per_digit: int):
        data = []
        labels = []
        
        for digit in range(10):
            for _ in range(samples_per_digit):
                img = self.create_digit_image(digit)
                data.append(img)
                labels.append(digit)
        
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data, labels = zip(*combined)
        
        return list(data), list(labels)
    
    def create_test_data(self, samples_per_digit: int):
        data = []
        labels = []
        
        for digit in range(10):
            for _ in range(samples_per_digit):
                img = self.create_digit_image(digit, add_noise=True)
                data.append(img)
                labels.append(digit)
        
        return data, labels
    
    def create_digit_image(self, digit: int, add_noise: bool = True):
        size = 20
        img = [0.0] * (size * size)
        
        if digit == 0:
            center = size // 2
            radius = size // 3
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - center)**2 + (j - center)**2)
                    if radius - 1 <= dist <= radius + 1:
                        img[i*size + j] = random.uniform(0.7, 0.9)
        elif digit == 1:
            col = size // 2
            for i in range(size // 4, size - size // 4):
                for offset in range(-1, 2):
                    if 0 <= i < size and 0 <= col + offset < size:
                        img[i*size + (col + offset)] = random.uniform(0.7, 0.9)
        elif digit == 2:
            for i in range(size):
                if i < size // 2:
                    img[i*size + (size - i - 2)] = random.uniform(0.7, 0.9)
                else:
                    img[i*size + (i - size//2)] = random.uniform(0.7, 0.9)
        elif digit == 3:
            for i in range(size):
                for j in range(size - 3, size):
                    img[i*size + j] = random.uniform(0.7, 0.9)
        elif digit == 4:
            for i in range(size):
                if i < size//2:
                    img[i*size + size//2] = random.uniform(0.7, 0.9)
                else:
                    for j in range(3, size-3):
                        img[i*size + j] = random.uniform(0.7, 0.9)
        elif digit == 5:
            for i in range(3):
                img[5*size + (5+i)] = random.uniform(0.7, 0.9)
                img[(10+i)*size + 5] = random.uniform(0.7, 0.9)
                img[15*size + (5+i)] = random.uniform(0.7, 0.9)
        elif digit == 6:
            for i in range(5, 15):
                if i == 5 or i == 14:
                    for j in range(5, 15):
                        img[i*size + j] = random.uniform(0.7, 0.9)
                else:
                    img[i*size + 5] = random.uniform(0.7, 0.9)
        elif digit == 7:
            for i in range(size):
                if i == 0:
                    for j in range(size):
                        img[i*size + j] = random.uniform(0.7, 0.9)
                else:
                    img[i*size + (size - i - 1)] = random.uniform(0.7, 0.9)
        elif digit == 8:
            for i in range(6, 14):
                if i == 6 or i == 13:
                    for j in range(6, 14):
                        img[i*size + j] = random.uniform(0.7, 0.9)
                else:
                    img[i*size + 6] = random.uniform(0.7, 0.9)
                    img[i*size + 13] = random.uniform(0.7, 0.9)
        elif digit == 9:
            for i in range(5, 15):
                if i == 5 or i == 14:
                    for j in range(5, 15):
                        img[i*size + j] = random.uniform(0.7, 0.9)
                else:
                    img[i*size + 14] = random.uniform(0.7, 0.9)
        
        if add_noise:
            for i in range(len(img)):
                if img[i] > 0.5:
                    img[i] += random.uniform(-0.1, 0.1)
                else:
                    img[i] = random.uniform(0, 0.2)
                img[i] = max(0, min(1, img[i]))
        
        if len(img) != 400:
            img = img[:400] if len(img) > 400 else img + [0.0] * (400 - len(img))
        
        return img
    
    @pyqtSlot()
    def enable_start_button(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    @pyqtSlot(float)
    def show_success_dialog(self, test_accuracy: float):
        msg = QMessageBox(self)
        msg.setWindowTitle("Обучение завершено")
        msg.setIcon(QMessageBox.Icon.Information)
        
        if test_accuracy > 85:
            msg.setText(f"✅ Отличный результат!\n\nТочность модели: {test_accuracy:.1f}%")
        elif test_accuracy > 70:
            msg.setText(f"⚠ Хороший результат!\n\nТочность модели: {test_accuracy:.1f}%")
        else:
            msg.setText(f"❌ Результат требует улучшения\n\nТочность модели: {test_accuracy:.1f}%")
        
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.button(QMessageBox.StandardButton.Yes).setText("Протестировать модель")
        msg.button(QMessageBox.StandardButton.No).setText("Вернуться в меню")
        
        reply = msg.exec()
        
        if reply == QMessageBox.StandardButton.Yes:
            self.go_back()
            self.main_window.open_testing()
        else:
            self.go_back()
    
    def stop_training_func(self):
        self.stop_training = True
        self.log("\n🛑 Запрос на остановку обучения...")
        self.stop_btn.setEnabled(False)
    
    def go_back(self):
        self.main_window.show()
        self.close()

# ==================== ОКНО ТЕСТИРОВАНИЯ ====================
class TestingWindow(QMainWindow):
    def __init__(self, model, main_window):
        super().__init__()
        self.model = model
        self.main_window = main_window
        
        self.init_ui()
        self.run_test()
    
    def init_ui(self):
        self.setWindowTitle("NEVMY - Тестирование модели")
        self.setGeometry(200, 200, 900, 700)
        
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #2c3e50, stop:1 #1a252f);
            }
        """)
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        title_label = QLabel("📊 Тестирование модели")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #e67e22;
                padding: 20px;
                background-color: rgba(230, 126, 34, 0.15);
                border-radius: 20px;
                border: 3px solid #e67e22;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 11))
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a252f;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        layout.addWidget(self.result_text, 1)
        
        self.back_btn = StableButton("← Назад", "#3498db")
        self.back_btn.clicked.connect(self.go_back)
        layout.addWidget(self.back_btn)
        
        self.setMinimumSize(800, 600)
    
    def log(self, message: str):
        self.result_text.append(message)
        self.result_text.verticalScrollBar().setValue(self.result_text.verticalScrollBar().maximum())
        QApplication.processEvents()
    
    def run_test(self):
        self.log("=" * 70)
        self.log(" " * 15 + "ТЕСТИРОВАНИЕ МОДЕЛИ")
        self.log("=" * 70)
        self.log(f"\n📊 Архитектура нейросети:")
        self.log(f"   • Входной слой: {self.model.input_size} нейронов")
        self.log(f"   • Скрытый слой: {self.model.hidden_size} нейронов")
        self.log(f"   • Выходной слой: {self.model.output_size} нейронов")
        self.log(f"   • Название модели: {self.model.model_name}")
        
        self.log("\n🧪 Создание тестовых данных...")
        test_data, test_labels = self.create_test_data(30)
        
        self.log(f"✅ Создано {len(test_data)} тестовых примеров")
        
        self.log("\n🔍 Запуск тестирования...")
        self.log("-" * 70)
        
        correct = 0
        confusion = [[0] * 10 for _ in range(10)]
        
        for i, (inputs, target) in enumerate(zip(test_data, test_labels)):
            try:
                probs = self.model.predict(inputs)
                predicted = probs.index(max(probs))
                
                confusion[target][predicted] += 1
                
                if predicted == target:
                    correct += 1
                
                if (i + 1) % 30 == 0:
                    self.log(f"   Обработано: {i+1}/{len(test_data)}")
                    
            except Exception as e:
                self.log(f"   Ошибка на примере {i}: {e}")
        
        accuracy = correct / len(test_data) * 100
        
        self.log("-" * 70)
        self.log(f"\n📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
        self.log(f"   • Правильно распознано: {correct}/{len(test_data)}")
        self.log(f"   • Точность: {accuracy:.1f}%")
        
        self.log(f"\n📊 Матрица ошибок:")
        header = "      " + " ".join(f"{i:^4}" for i in range(10))
        self.log(header)
        self.log("     " + "-" * 49)
        for i in range(10):
            row = f"{i:2} | " + " ".join(f"{confusion[i][j]:^4}" for j in range(10))
            self.log(row)
        
        self.log("\n" + "=" * 70)
        
        if accuracy > 85:
            self.log(" " * 10 + "✅ Модель работает ОТЛИЧНО!")
        elif accuracy > 70:
            self.log(" " * 10 + "⚠ Модель работает НОРМАЛЬНО")
        else:
            self.log(" " * 10 + "❌ Модель нуждается в ДООБУЧЕНИИ")
        
        self.log("=" * 70)
        
        Logger().log(f"Тестирование модели '{self.model.model_name}': точность {accuracy:.1f}%", "INFO")
    
    def create_test_data(self, samples_per_digit: int):
        data = []
        labels = []
        
        for digit in range(10):
            for _ in range(samples_per_digit):
                img = self.create_digit_image(digit)
                data.append(img)
                labels.append(digit)
        
        return data, labels
    
    def create_digit_image(self, digit: int):
        size = 20
        img = [0.0] * (size * size)
        
        if digit == 0:
            center = size // 2
            radius = size // 3
            for i in range(size):
                for j in range(size):
                    dist = math.sqrt((i - center)**2 + (j - center)**2)
                    if radius - 1 <= dist <= radius + 1:
                        img[i*size + j] = random.uniform(0.7, 0.9)
        else:
            for i in range(size // 3, 2 * size // 3):
                for j in range(size // 3, 2 * size // 3):
                    if random.random() > 0.7:
                        img[i*size + j] = random.uniform(0.6, 0.9)
        
        for i in range(len(img)):
            if img[i] > 0.5:
                img[i] += random.uniform(-0.1, 0.1)
            else:
                img[i] = random.uniform(0, 0.2)
            img[i] = max(0, min(1, img[i]))
        
        return img
    
    def go_back(self):
        self.main_window.show()
        self.close()

# ==================== ДИАЛОГ ЛОГОВ ====================
class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Логи программы")
        self.setGeometry(300, 300, 900, 600)
        
        layout = QVBoxLayout(self)
        
        control_layout = QHBoxLayout()
        self.date_combo = QComboBox()
        self.date_combo.setMinimumHeight(40)
        self.date_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495e;
                color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        self.date_combo.currentTextChanged.connect(self.load_log_file)
        
        self.refresh_btn = StableButton("🔄 Обновить", "#16a085")
        self.refresh_btn.clicked.connect(self.load_log_dates)
        
        control_layout.addWidget(QLabel("Дата логов:"))
        control_layout.addWidget(self.date_combo)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a252f;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.log_text)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
        """)
        layout.addWidget(button_box)
        
        self.load_log_dates()
    
    def load_log_dates(self):
        logger = Logger()
        log_dir = logger.get_log_dir()
        dates = []
        
        if os.path.exists(log_dir):
            for file in os.listdir(log_dir):
                if file.startswith("nevmy_") and file.endswith(".log"):
                    date_str = file[6:16]
                    dates.append(date_str)
        
        self.date_combo.clear()
        if dates:
            dates.sort(reverse=True)
            self.date_combo.addItems(dates)
            self.load_log_file(dates[0])
        else:
            self.date_combo.addItem("Логи не найдены")
            self.log_text.setText("Логи не найдены")
    
    def load_log_file(self, date_str):
        if date_str == "Логи не найдены":
            return
        
        try:
            logger = Logger()
            log_dir = logger.get_log_dir()
            log_file = os.path.join(log_dir, f"nevmy_{date_str}.log")
            
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.log_text.setText(content[-50000:])
                self.log_text.moveCursor(QTextCursor.MoveOperation.End)
            else:
                self.log_text.setText(f"Лог-файл не найден: {log_file}")
        except Exception as e:
            self.log_text.setText(f"Ошибка чтения логов: {str(e)}")

# ==================== ЗАПУСК ПРОГРАММЫ ====================
def main():
    # Инициализация logging
    logging.basicConfig(level=logging.INFO)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Устанавливаем стиль для всего приложения
    app.setStyleSheet("""
        * {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            font-weight: bold;
        }
        QLabel {
            font-weight: normal;
        }
    """)
    
    window = MainWindow()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()