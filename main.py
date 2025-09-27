from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from models import AnalysisTask, User, get_db, create_tables,create_task,update_task, delete_task_from_db
from datetime import datetime
import os
import json
from pathlib import Path
import uuid
import mistune
import json
import requests

# Your previous functions here...
from worker import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
)

from worker import submit_pipeline

app = FastAPI(title="WiseHire Technology")

# Create directories
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)  # 确保temp目录存在
Path("analysis_reports").mkdir(exist_ok=True)  # 确保analysis_reports目录存在

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create database tables
create_tables()

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, username: str = None, db: Session = Depends(get_db)):
    if not username:
        return RedirectResponse("/")
    
    # 验证用户是否存在
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return RedirectResponse("/")
    
    # Get user's analysis tasks
    user_tasks = db.query(AnalysisTask).filter(AnalysisTask.user_id == user.id).order_by(AnalysisTask.submitted_at.desc()).all()
    
    # 转换任务数据为字典格式，便于前端处理
    tasks_data = []
    for task in user_tasks:
        tasks_data.append({
            "id": task.id,
            "resume_filename": task.resume_filename,
            "job_description": task.job_description[:50] + "..." if len(task.job_description) > 50 else task.job_description,
            "submitted_at": task.submitted_at.strftime('%Y-%m-%d %H:%M'),
            "status": task.status,
            "result_json": task.result_json,
            "updated_at": task.updated_at.strftime('%Y-%m-%d %H:%M') if task.updated_at else None
        })
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username,
        "tasks": user_tasks,
        "tasks_json": json.dumps(tasks_data)  # 传递JSON格式的任务数据
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, username: str = None, db: Session = Depends(get_db)):
    if not username:
        return RedirectResponse("/")
    
    # 验证用户是否存在
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return RedirectResponse("/")
    
    return templates.TemplateResponse("upload.html", {"request": request, "username": username})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # 验证用户凭据
    user = db.query(User).filter(User.username == username, User.password == password).first()
    
    if user:
        return RedirectResponse(f"/dashboard?username={username}", status_code=303)
    else:
        return templates.TemplateResponse("login.html", {
            "request": Request(scope={"type": "http"}),
            "error": "Invalid credentials"
        })

# 新增API：获取任务状态
@app.get("/api/tasks/{username}")
async def get_user_tasks(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return JSONResponse({"error": "User not found"}, status_code=404)
    
    tasks = db.query(AnalysisTask).filter(AnalysisTask.user_id == user.id).order_by(AnalysisTask.submitted_at.desc()).all()
    
    tasks_data = []
    for task in tasks:
        tasks_data.append({
            "id": task.id,
            "resume_filename": task.resume_filename,
            "job_description": task.job_description[:50] + "..." if len(task.job_description) > 50 else task.job_description,
            "submitted_at": task.submitted_at.strftime('%Y-%m-%d %H:%M'),
            "status": task.status,
            "result_json": task.result_json[:100] if task.result_json else "",
            "updated_at": task.updated_at.strftime('%Y-%m-%d %H:%M') if task.updated_at else None
        })
    
    return {"tasks": tasks_data}


@app.get("/result/{task_id}")
async def result_page(request: Request, task_id: int, username: str, db: Session = Depends(get_db)):
    # 验证用户权限
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return RedirectResponse("/")
    
    task = db.query(AnalysisTask).filter(AnalysisTask.id == task_id, AnalysisTask.user_id == user.id).first()
    if not task:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Task not found"
        })
    
    if task.status != 'completed' or not task.result_json:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Analysis result not available"
        })
    
    # 解析JSON结果
    result_data = json.load(open(task.result_json))

    # 美化JSON用于显示
    temp_his=[]
    for idx in range(len(result_data["detailed_work_histories"])):
        temp_his.append(json.loads(result_data["detailed_work_histories"][idx]))

    result_data["detailed_work_histories"]=temp_his
    
    from markupsafe import Markup
    formatted_json = json.dumps(result_data, indent=2, ensure_ascii=False)
    formatted_json = Markup(formatted_json)  # 标记为安全的HTML

    data={
        "request": request,
        "username": username,
        "task": task,
        "result_data": result_data,
        "formatted_json": formatted_json
    }

    return templates.TemplateResponse("result.html", data)

@app.post("/analyze")
async def analyze_resume(
    request: Request,
    file: UploadFile = File(...),  # 使用 ... 表示必填字段
    job_description: str = Form(...),  # 使用 ... 表示必填字段
    username: str = Form(...),
    db: Session = Depends(get_db)
):
    # 验证用户
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    user_id = user.id
    
    # 验证文件类型
    allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Please provide: {', '.join(allowed_extensions)} documents"
        )
    
    # 验证文件内容
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Resume file cannot be empty")
    
    # 重置文件指针以便后续使用
    await file.seek(0)
    
    # 验证职位描述
    job_description = job_description.strip()
    if len(job_description.split()) < 10:
        raise HTTPException(status_code=400, detail="Job description must be at least 10 words long")
    
    # 创建任务记录
    task_data = {
        "user_id": user_id,
        "resume_filename": file.filename, 
        "job_description": job_description, 
        "submitted_at": datetime.now(), 
        "status": "in_process",
        "result_json": "",
        "updated_at": datetime.now()
    }

    task_id = create_task(task_data)

    try:
        # 保存文件
        file_extension = file.filename.split(".")[-1].lower()
        file_data = await file.read()
        
        # 确保文件扩展名有效
        if file_extension not in ['pdf', 'docx', 'txt']:
            raise HTTPException(status_code=400, detail="File must be PDF, DOCX, or TXT format")

        # 保存上传的文件
        file_path = os.path.join("static/uploads",f"task_id_{task_id}_user_{user_id}.{file_extension}")


        with open(file_path, "wb") as f:
            f.write(file_data)

        # 提取简历文本
        if file_extension == "pdf":
            resume_text = extract_text_from_pdf(file_data)
        elif file_extension == "docx":
            resume_text = extract_text_from_docx(file_path)
        else:  # txt
            resume_text = extract_text_from_txt(file_path)

        submit_pipeline.delay(task_id,user_id,resume_text,job_description)

        return JSONResponse(
            status_code=200,
            content={"message": "Task submitted successfully"}
        )
    except Exception as e:
        # 如果分析过程中出现错误，更新任务状态为失败
        update_data = {
            "task_id": task_id,
            "status": "failed",
            "updated_at": datetime.now()
        }
        update_task(update_data)
        
        print(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.delete("/api/tasks/{task_id}")
async def delete_task(
    task_id: int, 
    username: str, 
    db: Session = Depends(get_db)
):
    """
    删除分析任务 - 使用psycopg2直接操作数据库
    """
        # 验证用户
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user.id

    # 调用删除函数
    try:
        success = delete_task_from_db(task_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or already deleted")
        
        return {
            "success": True,
            "message": "Task deleted successfully",
            "deleted_task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)