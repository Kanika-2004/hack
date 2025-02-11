const {loginUser,logout}=require('../controllers/authController')
const express=require('express')
const authrouter=express.Router()
authrouter.post('/login',loginUser)
authrouter.post('/logout',logout)
module.exports=authrouter