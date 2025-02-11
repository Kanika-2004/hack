const serverConfig = require('../config/serverConfig');
const login=require('../services/authService')

async function logout(req,res){
    console.log(res.cookie)
    console.log(req.cookie)
    res.cookie("authToken","",{
        httpOnly:true,
        sameSite: "lax",
        secure: serverConfig.COOKIE_SECURE,
        // maxAge: 7 * 24 * 60 * 60 * 1000,
        // domain:serverConfig.FRONTEND_URL
    })
     return res.status(200).json({
        success: true,
        message: "Log out successfull",
        error: {},
        data: {
            isLoggedIn: false  // ✅ Send this to the frontend
        }
    });
}


async function loginUser(req,res){
    try{
        const response=await login(req.body)
        // const cookievalue= JSON.stringify(response)
        res.cookie("authToken",response,{
            httpOnly:true,
            sameSite:'lax',
            secure:serverConfig.COOKIE_SECURE,
            // maxAge:7*24*60*60*1000,
            // domain:serverConfig.FRONTEND_URL
        });
        return res.status(201).json({
            data:{
                // userData:response.userData
            },
            success:true,
            error:{},
            message:"successfully logged in the user",
            data: {
                isLoggedIn: true  // ✅ Send this to the frontend
            }
        })
    }catch(error){
        console.log(error);
        return res.status(402).json({
            success:false,
            message:"an error occurred"
        })
    }

}
module.exports={
    loginUser,
    logout
}