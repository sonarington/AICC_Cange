import express from "express"; import bodyParser from "body-parser"; import path from "path"; import { fileURLToPath } from "url"; import { spawn } from "child_process"; import multer from "multer"; import cors from "cors"; import axios from "axios"; import { createServer } from "http"; import { Server } from "socket.io"; import { profile } from "console";  import mysql from "mysql2/promise";  import session from "express-session";  import passport from "passport"; import { Strategy as LocalStrategy } from "passport-local"; import bcrypt from "bcryptjs";   import jwt from "jsonwebtoken"; const secretKey = "your-secret-key";   import dotenv from "dotenv"; dotenv.config();   const __filename = fileURLToPath(import.meta.url); const __dirname = path.dirname(__filename);   const app = express(); const port = 3333;   const pool = mysql.createPool({ host: "morontia.com", user: "test", password: "asdf1234", database: "cange", });  const server = createServer(app); const io = new Server(server, { cors: {  origin: true,  methods: ["GET", "POST"], credentials: true, }, });  const corsOptions = { origin: true,   credentials: true, };  app.use(cors(corsOptions));     app.use(express.static(__dirname + "/dist"));   app.use(express.json());  const sessionMiddleware = session({  secret: "your-secret-key", resave: false,  saveUninitialized: false, cookie: { httpOnly: true, sameSite: "None",  secure: false,  maxAge: 3600000, }, });   app.use(sessionMiddleware);  io.use((socket, next) => { sessionMiddleware(socket.request, {}, next); });  app.use(passport.initialize()); app.use(passport.session());   passport.use( new LocalStrategy( { usernameField: "user_id", passwordField: "password", }, async (username, password, done) => { try { console.log("Received:", username, password);  const [results] = await pool.query( "SELECT * FROM users WHERE user_id = ?", [username] );    if (results.length === 0) { console.log("No user found"); return done(null, false, { message: "Incorrect username." }); }  const user = results[0]; console.log("User found:", user);  const match = await bcrypt.compare(password, user.password_hash); if (match) { console.log("Password match"); return done(null, user); } console.log("Incorrect password"); return done(null, false, { message: "Incorrect password." }); } catch (err) { console.error("Error:", err); return done(err); } } ) );  passport.serializeUser((user, done) => { done(null, user.id); });      passport.deserializeUser((id, done) => { pool.query("SELECT * FROM users WHERE id = ?", [id], (err, results) => { if (err) return done(err); done(null, results[0]); }); });   app.post("/session-login", (req, res, next) => { console.log("Login request received:", req.body); console.log("Current sessionID:", req.sessionID); console.log("Current session:", req.session);  passport.authenticate("local", (err, user, info) => { if (err) { console.log(err); return next(err); } if (!user) { console.log(info); return res.status(401).json({ message: info.message }); } req.logIn(user, (err) => { if (err) return next(err);   console.log("login sessionID:", req.sessionID); console.log("login session:", req.session);    req.session.userId = user.id;   req.session.save((err) => { if (err) return next(err); res.json({ message: "Login successful", nickname: user.nickname, user: req.user, }); });  console.log("Current sessionID:", req.sessionID); console.log("Current session:", req.session); console.log("isAuthenticated:", req.isAuthenticated());   }); })(req, res, next); });   app.get("/chatRooms", async (req, res) => { try { const [results] = await pool.query("SELECT * FROM chat_rooms");   if (results.length === 0) { return res.status(404).json({ message: "채팅룸을 찾을 수 없습니다." }); }   res.status(200).json(results); } catch (err) { console.error("Error:", err); return res.status(500).json({ message: "서버 오류" }); } });   app.post("/createChatRooms", async (req, res) => { console.log(req.body); const { chatRoomName, chatRoomSub } = req.body;  try {  await pool.query("INSERT INTO chat_rooms (name, sub) VALUES (?, ?)", [ chatRoomName, chatRoomSub, ]);   res.status(201).json({ message: "chat_rooms registered successfully" });  } catch (error) { console.error(error); return res.status(500).json({ message: "서버 오류" });  } });   const verifyToken = (req, res, next) => { const token = req.headers["authorization"];  if (!token) return res.status(403).send("Token is required");  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => { if (err) return res.status(401).send("Invalid token"); req.userId = decoded.id; next(); }); };   app.get("/protected", verifyToken, (req, res) => { res.send("This is a protected route"); });  app.get("/verify", async (req, res) => { const token = req.headers["authorization"]?.split(" ")[1];  if (!token) { return res.status(401).json({ message: "토큰이 없습니다." }); }  try { const decoded = jwt.verify(token, process.env.JWT_SECRET); const userId = decoded.id;    const [results] = await pool.query( "SELECT nickname FROM users WHERE id = ?", [userId] );  const nickname = results[0].nickname;  res.status(200).json({ message: "사용자가 로그인 되어 있습니다.", nickname: nickname, user: decoded, }); } catch (err) { console.error("Error:", err); return res.status(500).json({ message: "서버 오류" }); } });   app.post("/logout", (req, res) => { const token = req.headers["authorization"]?.split(" ")[1];  res.status(200).json({ message: "Logout successful" }); });   app.put("/change-nickname", async (req, res) => { const token = req.headers["authorization"]?.split(" ")[1];  if (!token) { return res.status(401).json({ message: "토큰이 없습니다." }); }  try { const decoded = jwt.verify(token, process.env.JWT_SECRET); const userId = decoded.id; const { new_nickname } = req.body;   const [nicknameCheck] = await pool.query( "SELECT * FROM users WHERE nickname = ?", [new_nickname] ); if (nicknameCheck.length > 0) { return res .status(409) .json({ message: `${new_nickname} 이 닉네임은 이미 사용 중입니다.` }); }   const [result] = await pool.query( "UPDATE users SET nickname = ? WHERE id = ?", [new_nickname, userId] );  if (result.affectedRows === 0) { return res.status(404).json({ message: "사용자를 찾을 수 없습니다." }); }  res.status(200).json({ message: "닉네임이 변경되었습니다.", new_nickname: new_nickname, user: decoded, }); } catch (err) { console.error("Error:", err); return res.status(500).json({ message: "서버 오류" }); } });   app.put("/check-nickname", async (req, res) => { try { const { new_nickname } = req.body;    const [nicknameCheck] = await pool.query( "SELECT * FROM users WHERE BINARY nickname = ?", [new_nickname] );  if (nicknameCheck.length > 0) {  return res .status(409) .json({ message: `이 닉네임은 이미 사용 중입니다.` }); }   res.status(200).json({ message: "사용가능한 닉네임입니다." }); } catch (err) { console.error("Error:", err); return res.status(500).json({ message: "서버 오류" }); } });   app.post("/login", async (req, res) => { console.log("Login request received:", req.body); const { user_id, password } = req.body;  try {  const [results] = await pool.query( "SELECT * FROM users WHERE user_id = ?", [user_id] );  if (results.length === 0) { console.log("User not found");  return res.status(401).json({ message: "User not found" }); }  const user = results[0];   if (!bcrypt.compareSync(password, user.password_hash)) { console.log("Invalid password");  return res.status(401).json({ message: "Invalid password" }); }   const token = jwt.sign({ id: user.id }, process.env.JWT_SECRET, { expiresIn: "1h", }); console.log("user.id:", user.id); res.json({ token: token, message: "Login successful", nickname: user.nickname, }); } catch (err) { console.log("Server error:", err);  return res.status(401).json({ message: "Server error" }); } });   const ensureAuthenticated = (req, res, next) => { if (req.isAuthenticated()) { return next(); } res.status(401).send("로그인이 필요합니다."); };   app.get("/api/check-auth", (req, res) => { console.log("check sessionID:", req.sessionID); console.log("check session:", req.session); if (req.isAuthenticated()) {  res.json({ authenticated: true, user: req.user }); console.log("check:", "true"); } else {  res.json({ authenticated: false }); console.log("check:", "false"); } });   app.get("/check", (req, res) => { console.log("check sessionID:", req.sessionID); console.log("check session:", req.session); if (req.isAuthenticated()) {  res.json({ authenticated: true, user: req.user }); console.log("check:", "true"); } else {  res.json({ authenticated: false, user: req.user }); console.log("check:", "false"); } });   app.get("/profile", ensureAuthenticated, (req, res) => {  res.send(`안녕하세요 ${req.user.username}님, 프로필 페이지입니다.`); });  app.get("/dashboard", ensureAuthenticated, (req, res) => { res.send("Welcome to your dashboard!"); });  app.post("/settings", ensureAuthenticated, (req, res) => { res.send("Settings updated!"); });  app.get("/protected", ensureAuthenticated, (req, res) => { res.send("This is a protected route"); });   app.get("/session-logout", (req, res) => { req.logout((err) => { if (err) return next(err); console.log("logout sessionID:", req.sessionID); console.log("logout session:", req.session); res.json({ message: "Logout successful" }); }); });   app.post("/signup", async (req, res) => { console.log(req.body); const { user_id, nickname, email, password } = req.body;  try {  const [rows] = await pool.query("SELECT * FROM users WHERE user_id = ?", [ user_id, ]);  if (rows.length > 0) { return res.status(400).send("Username already exists"); }   const hashedPassword = await bcrypt.hash(password, 10);   await pool.query( "INSERT INTO users (user_id, nickname, email, password_hash) VALUES (?, ?, ?, ?)", [user_id, nickname, email, hashedPassword] );  res.status(201).send("User registered successfully");  } catch (error) { console.error(error); res.status(500).send("Internal server error"); } });   const storage = multer.memoryStorage(); const upload = multer({ storage: storage });   app.post("/upload", (req, res) => { const base64Image = req.body.image; const filePath = path.join(__dirname, "uploads", "capture.png");  ensureDirectoryExistence(filePath);  const imageBuffer = Buffer.from(base64Image, "base64"); fs.writeFile(filePath, imageBuffer, (err) => { if (err) { console.error("Error saving image:", err); return res .status(500) .json({ success: false, message: "Error saving image" }); } console.log("Image saved successfully"); res.json({ success: true, message: "Image saved successfully" }); }); });             app.post("/analyze", upload.single("file"), (req, res) => { if (!req.file) { console.log("No file uploaded"); return res.status(400).json({ error: "No file uploaded" }); }  console.log("File uploaded:", req.file);   const pythonProcess = spawn("python", ["cpo.py"]);  pythonProcess.stdin.write(req.file.buffer); pythonProcess.stdin.end();  let result = ""; pythonProcess.stdout.on("data", (data) => {  const lines = data.toString().split("\n"); lines.forEach((line) => { try { const parsedline = JSON.parse(line); result += JSON.stringify(parsedline); } catch (e) {} }); console.log("current result data:", result); });      pythonProcess.stdout.on("end", () => { try { console.log("result data:", result); const colorresult = JSON.parse(result); if (!res.headersSent) { res.json(colorresult); } } catch (error) { console.error("Failed to parse analysis result: ", error); console.error("result data received:", result); if (!res.headersSent) { res.status(500).json({ error: "Failed to parse analysis result" }); } } });  pythonProcess.stderr.on("data", (data) => { console.error(`python script error: ${data}`); res.status(500).json({ error: "color analysis failed" }); });        });   const executePythonScript = (scriptPath, args, res) => { const pythonProcess = spawn("python", [scriptPath, ...args]);  let result = ""; pythonProcess.stdout.on("data", (data) => { result += data.toString(); });  pythonProcess.stderr.on("data", (data) => { console.error(`stderr: ${data}`); if (!res.headersSent) { res.status(500).json({ error: data.toString() }); } });  pythonProcess.on("close", (code) => { if (!res.headersSent) { res.json({ result }); } }); };   app.post("/calculate", (req, res) => { const { expression } = req.body; res.setHeader("Content-Type", "application/json; charset=utf-8"); executePythonScript("fpo.py", [expression], res); });   app.get("/", (req, res) => { res.sendFile(__dirname + "/dist/index.html"); });   app.post("/process_message", async (req, res) => { console.log("received request:", req.body); try { const { text } = req.body; const response = await axios.post("http://localhost:5000/process_message", { text, }); res.json(response.data); } catch (error) { console.error("error processing text:", error.message); res.status(500).json({ error: "failed to process text" }); } });   app.get("/messages", (req, res) => { res.json([{ type: "received", username: `${users[socket.id]}` }]); });   app.post("/analyze-face", upload.single("file"), (req, res) => {  if (!req.file) { console.log("No file uploaded"); return res.status(400).json({ error: "No file uploaded" }); }  console.log("File uploaded:", req.file);   const python_process = spawn("python", ["face_landmark_detection.py"]);  python_process.stdin.write(req.file.buffer); python_process.stdin.end();  let resultData = ""; python_process.stdout.on("data", (data) => { resultData += data.toString(); });  python_process.stdout.on("end", () => { try { const analysisResult = JSON.parse(resultData); if (!res.headersSent) { res.json(analysisResult); } } catch (error) { console.error("Faild to parse analysis result:", error); if (!res.headersSent) { res.status(500).json({ error: "Failed to parse analysis result" }); } } });  python_process.stderr.on("data", (data) => { console.error(`Python script error: ${data}`); res.status(500).json({ error: "Face analysis failed" }); }); });   app.post("/analyze-face2", upload.single("file"), (req, res) => { if (!req.file) { console.log("No file uploaded"); return res.status(400).json({ error: "No file uploaded" }); }  console.log("File uploaded:", req.file);   const python_process = spawn("python", ["face_model_detection.py"]);  let resultData = ""; let errorOccurred = false;   python_process.stdin.write(req.file.buffer); python_process.stdin.end();  python_process.stdout.on("data", (data) => {  if (errorOccurred) return; const lines = data.toString().split("\n"); lines.forEach((line) => {  try { const parsedLine = JSON.parse(line); resultData += JSON.stringify(parsedLine); } catch (e) {} }); console.log("current result data:", resultData); });  python_process.stdout.on("end", () => { if (errorOccurred) return; try { console.log("result data:", resultData); const analysisResult = JSON.parse(resultData); if (!res.headersSent) { res.json(analysisResult); } } catch (error) { console.error("Failed to parse analysis result:", error); console.error("result data received:", resultData); if (!res.headersSent) { res.status(500).json({ error: "Failed to parse analysis result" }); } } });  python_process.stderr.on("data", (data) => { console.error(`Python script error: ${data}`); if (!res.headersSent) { errorOccurred = true; res.status(500).json({ error: "Face analysis failed" }); } }); });   app.post("/faceLandmark_blob", upload.single("image"), (req, res) => { if (!req.file) { return res.status(400).send("No file uploaded."); }  const fileBuffer = req.file.buffer; const pythonProcess = spawn("python", ["./faceLandmark_blob.py"], { stdio: ["pipe", "pipe", "pipe"], });  pythonProcess.stdin.write(fileBuffer); pythonProcess.stdin.end();  let result = ""; pythonProcess.stdout.on("data", (data) => { result += data.toString(); });  pythonProcess.stderr.on("data", (data) => { console.error(`Python error: ${data}`); });  pythonProcess.on("close", (code) => { if (code === 0) { try { const jsonResult = JSON.parse(result); const { face_shape, image_base64 } = jsonResult; if (!image_base64) { throw new Error("Image field is missing in the JSON output"); } res.json({ face_shape, image_base64 }); } catch (error) { res.status(500).json({ error: "Invalid JSON output" }); } } else { res.status(500).json({ error: "Python script failed" }); } }); });   app.post("/faceLandmark_path", (req, res) => { const { expression } = req.body; const pythonProcess = spawn("python", ["./faceLandmark_path.py", expression]);  let result = ""; pythonProcess.stdout.on("data", (data) => { result += data.toString(); });  pythonProcess.stderr.on("data", (data) => { console.error(`stderr: ${data}`); if (!res.headersSent) { res.status(500).json({ error: data.toString() }); } });  pythonProcess.on("close", (code) => { if (!res.headersSent) { res.json({ result }); } }); });   app.post("/personalColor_blob", upload.single("image"), (req, res) => { if (!req.file) { return res.status(400).send("No file uploaded."); }  const fileBuffer = req.file.buffer; const pythonProcess = spawn("python", ["./personalColor_blob.py"], { stdio: ["pipe", "pipe", "pipe"], });  pythonProcess.stdin.write(fileBuffer); pythonProcess.stdin.end();  let result = ""; pythonProcess.stdout.on("data", (data) => { result += data.toString(); });  pythonProcess.stderr.on("data", (data) => { console.error(`stderr: ${data}`); if (!res.headersSent) { res.status(500).json({ error: data.toString() }); } });  pythonProcess.on("close", (code) => { if (!res.headersSent) { res.json({ result }); } }); });   app.post("/personalColor_path", (req, res) => { const { expression } = req.body; const pythonProcess = spawn("python", [ "./personalColor_path.py", expression, ]);  let result = ""; pythonProcess.stdout.on("data", (data) => { result += data.toString(); });  pythonProcess.stderr.on("data", (data) => { console.error(`stderr: ${data}`); if (!res.headersSent) { res.status(500).json({ error: data.toString() }); } });  pythonProcess.on("close", (code) => { if (!res.headersSent) { res.json({ result }); } }); });   const users = {};   io.on("connection", (socket) => { console.log("a user connected:", socket.id);     socket.on("setUserInfo", (userInfo) => {  users[socket.id] = { nickname: userInfo.nickname, profile_photo: userInfo.profile_photo,   connectionTime: new Date(), };  console.log(`User ${userInfo.username} connected with ID: ${socket.id}`);   io.emit("userConnected", { id: socket.id, ...users[socket.id], }); });   socket.on("setUsername", (nickname) => {  users[socket.id] = { ...users[socket.id], nickname: nickname, };   io.emit("userConnected", { nickname, id: socket.id }); });   socket.on("updateUserProfilePhoto", async (imageUrl) => {  console.log( `User "${ users[socket.id].nickname }" changed "${"profile_photo"}" with ID: "${socket.id}"` );  users[socket.id] = { ...users[socket.id], profile_photo: imageUrl, };  io.emit("userChangePhoto", { id: socket.id }); });   socket.on("updateUsername", async (nickname) => {  console.log( `User "${users[socket.id].nickname}" changed "${nickname}" with ID: "${ socket.id }"` );  users[socket.id] = { ...users[socket.id], nickname: nickname, };   io.emit("userConnected", { nickname, id: socket.id }); });   socket.on("getRoomSize", (room) => { const roomSize = io.sockets.adapter.rooms.get(room)?.size || 0; socket.emit("roomSizeInfo", { room, roomSize }); });   socket.on("getAllRoomsInfo", () => { const rooms = io.sockets.adapter.rooms; const roomsInfo = [];  rooms.forEach((room, roomId) => { if (!rooms.has(roomId)) return; const roomSize = room.size; roomsInfo.push({ roomId, roomSize }); });  socket.emit("allRoomsInfo", roomsInfo); });   socket.on("joinRoom", (room) => { socket.join(room); console.log(`${socket.id} joined room: ${room}`); socket .to(room) .emit("notice", `${users[socket.id].nickname} 님이 들어왔습니다.`); socket.emit("notice", `${room} 채팅방에 입장하셨습니다.`); const roomSize = io.sockets.adapter.rooms.get(room)?.size || 0; console.log(`Room ${room} has ${roomSize} users`); io.to(room).emit("roomUsers", roomSize); });   socket.on("leaveRoom", (room) => { socket.leave(room); console.log( `${users[socket.id].nickname} 님이 ${room} 채팅방을 퇴장하셨습니다.` ); socket.emit("notice", `${room} 채팅방을 퇴장하셨습니다.`); const roomSize = io.sockets.adapter.rooms.get(room)?.size || 0; console.log(`Room ${room} now has ${roomSize} users`); io.to(room).emit("roomUsers", roomSize); });   socket.on("chatMessage", ({ room, message }) => { console.log(`Message from ${socket.id} in room ${room}: ${message}`); socket.to(room).emit("message", {  nickname: users[socket.id].nickname, profile_photo: users[socket.id].profile_photo, text: message,  }); });  socket.on("imageUpload", ({ room, imageUrl }) => {  console.log(`${"imageUpload"}`); socket.to(room).emit("newImage", { imageUrl }); });   socket.on("disconnect", () => { console.log("user disconnected:", socket.id); const username = users[socket.id]; if (username) { delete users[socket.id]; io.emit("userDisconnected", { username, id: socket.id }); } }); });  server.listen(port, () => { console.log(`Server listening on port ${port}`); });