from src.database.storage import DatabaseManager
import numpy as np

db = DatabaseManager()
db.initialize()

left_template = np.random.rand(4096).astype(np.float32)
right_template = np.random.rand(4096).astype(np.float32)

db.save_iris_template("user1", left_template, "left")
db.save_iris_template("user1", right_template, "right")

print("Left:", db.get_iris_template("user1", "left")[:10])
print("Right:", db.get_iris_template("user1", "right")[:10])