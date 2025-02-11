const jwt=require('jsonwebtoken')
const { JWT_SECRET } = require('../config/serverConfig')
const serverConfig = require('../config/serverConfig')
async function isLoggedIn(req,res,next){
    console.log("inside isloggedin the cookies are",req.cookies)
    // console.log(req.cookies)
    const token=req.cookies["authToken"]
    console.log("The token is",token)
    if(!token){
        return res.status(401).json({
            success:false,
            data:{},
            message:"no token provided by user"        
        })
    }
    try{
        const decoded=jwt.verify(token,JWT_SECRET)
        // console.log(decoded)
        console.log(decoded, decoded.exp, Date.now() / 1000);
        if(!decoded){
            throw{message:"wrong token provided"}
        
        }
        req.user={
            email:decoded.email,
            id:decoded.id
        }
        console.log(req.user);
        next();
    }catch(error){
        console.log(error);
        console.log(error.name);
        if(error.name === "TokenExpiredError") {
            res.cookie("authToken","",{
                httpOnly:true,
                secure:false,
                // maxAge: 7 * 30* 24 * 60 * 60 * 1000,
                                
            })
             return res.status(200).json({
                success: true,
                message: "Log out successfull ",
                error: {},
                data: {}
            });
        }

        return res.status(401).json({
            success:false,
            message:"an error occurred"
        })
    }
}
module.exports=isLoggedIn