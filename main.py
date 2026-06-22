import uvicorn
import os


def main():
    print("Hello from medical-assisstant!")


if __name__ == "__main__":
    main()


if _name_ == "_main_":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )