# QR Processing API - Service Management Guide

## 🎯 **Production Deployment (Systemd Service)**

The QR Processing API is now configured as a systemd service for production deployment with automatic startup, restart capabilities, and proper logging.

## 🚀 **Service Setup (Already Completed)**

The service has been set up with the following configuration:

```bash
# Service file location
/etc/systemd/system/qr-processing-api.service

# Service configuration
- User: ubuntu
- Working Directory: /home/ubuntu/workflows/qr_code_app
- Python Path: /opt/conda/bin/python
- Auto-restart on failure
- Loads environment variables from .env file
```

## 📋 **Service Management Commands**

### **Check Service Status**
```bash
sudo systemctl status qr-processing-api
```

### **Start/Stop/Restart Service**
```bash
# Start the service
sudo systemctl start qr-processing-api

# Stop the service
sudo systemctl stop qr-processing-api

# Restart the service (after code changes)
sudo systemctl restart qr-processing-api
```

### **Enable/Disable Auto-Start**
```bash
# Enable auto-start on boot (already enabled)
sudo systemctl enable qr-processing-api

# Disable auto-start
sudo systemctl disable qr-processing-api
```

## 📊 **Monitoring & Logs**

### **View Real-time Logs**
```bash
# Follow logs in real-time
sudo journalctl -u qr-processing-api -f

# View last 50 log entries
sudo journalctl -u qr-processing-api -n 50

# View logs from today
sudo journalctl -u qr-processing-api --since today

# View logs with timestamps
sudo journalctl -u qr-processing-api -f --no-pager
```

### **Check Service Health**
```bash
# Quick health check
curl -X GET "http://localhost:8002/health"

# External health check
curl -X GET "http://13.48.174.160:8002/health"

# Check if service is listening on port
sudo netstat -tlnp | grep 8002
# or
sudo ss -tlnp | grep 8002
```

## 🔄 **Making Code Changes**

When you modify the QR processing API code:

1. **Edit your Python files** (service runs from /home/ubuntu/workflows/qr_code_app)
2. **Restart the service** to apply changes:
   ```bash
   sudo systemctl restart qr-processing-api
   ```
3. **Check logs** to verify restart:
   ```bash
   sudo journalctl -u qr-processing-api -f --no-pager -n 20
   ```
4. **Test functionality**:
   ```bash
   curl -X GET "http://localhost:8002/health"
   ```

## 🛠️ **Troubleshooting**

### **Service Won't Start**
```bash
# Check service status for errors
sudo systemctl status qr-processing-api

# View detailed error logs
sudo journalctl -u qr-processing-api -n 50

# Check if port is already in use
sudo ss -tlnp | grep 8002

# Manually test the script
cd /home/ubuntu/workflows/qr_code_app
/opt/conda/bin/python qr_processing_api.py
```

### **Database Connection Issues**
```bash
# Check if environment variables are loaded
sudo systemctl show qr-processing-api --property=Environment

# Test database connection manually
# (Check logs for Supabase connection errors)
sudo journalctl -u qr-processing-api | grep -i supabase
```

### **Permission Issues**
```bash
# Check file permissions
ls -la /home/ubuntu/workflows/qr_code_app/qr_processing_api.py

# Check service file permissions
ls -la /etc/systemd/system/qr-processing-api.service

# Reload systemd if service file was modified
sudo systemctl daemon-reload
```

## 📈 **Performance Monitoring**

### **Resource Usage**
```bash
# Check memory and CPU usage
sudo systemctl status qr-processing-api

# Detailed process information
ps aux | grep qr_processing_api

# Monitor in real-time
top -p $(pgrep -f qr_processing_api.py)
```

### **API Performance**
```bash
# Test response time
time curl -X GET "http://localhost:8002/health"

# Test folder listing performance
time curl -X GET "http://localhost:8002/s3-folders"

# Monitor request logs
sudo journalctl -u qr-processing-api -f | grep "GET\|POST"
```

## 🔧 **Service Configuration Details**

The systemd service is configured with:

- **Automatic Restart**: Service restarts if it crashes
- **Environment File**: Loads variables from `/home/ubuntu/workflows/.env`
- **Security**: Runs with limited privileges
- **Logging**: All output goes to systemd journal
- **Resource Limits**: File handle limits set to 65536
- **Timeouts**: 60s start timeout, 30s stop timeout

## 🚀 **Production Benefits**

Running as a systemd service provides:

- ✅ **Automatic startup** on server boot
- ✅ **Automatic restart** on crashes
- ✅ **Centralized logging** via journald
- ✅ **Process management** by systemd
- ✅ **Security isolation** with user/group restrictions
- ✅ **Resource monitoring** and limits
- ✅ **Easy management** with standard systemctl commands
- ✅ **Fresh signed URLs** automatically generated for cached results (prevents URL expiration)

## 📋 **Daily Operations Checklist**

### **Health Check**
```bash
# 1. Check service status
sudo systemctl is-active qr-processing-api

# 2. Test API endpoint
curl -X GET "http://13.48.174.160:8002/health"

# 3. Check recent logs for errors
sudo journalctl -u qr-processing-api --since "1 hour ago" | grep -i error
```

### **After Code Updates**
```bash
# 1. Restart service
sudo systemctl restart qr-processing-api

# 2. Verify startup
sudo systemctl status qr-processing-api

# 3. Test functionality
curl -X GET "http://13.48.174.160:8002/s3-folders" | head -5
```

The QR Processing API is now fully production-ready with enterprise-grade service management! 🚀
