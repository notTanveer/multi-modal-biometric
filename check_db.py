from src.database.storage import DatabaseManager

db = DatabaseManager()
db.initialize()

users = db.list_users()

print("Enrolled Users:")
for user in users:
    print(user)