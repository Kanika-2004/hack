const mongoose=require('mongoose')
const bcrypt=require('bcrypt')
const Userschema= new mongoose.Schema({
    firstName:{
        type:String,
        // minlength:[5,"Name should be minimum of 5 character long"],
        lowercase:true,
        trim:true,
        // maxlength:[20,"FirstName should be of max length 20"]
    },
    lastName:{
        type:String,
        // minlength:[5,"LastName should be minimum of 5 character long"],
        lowercase:true,
        trim:true,
        // maxlength:[20,"LastName should be of max length 20"]
    },
   
    mobilenumber:{
        type:String,
       
        // maxlength:[10,"Mobile number should be 10 character long"],
        // minlength:[10,"Mobile number shoudl be 10 character long"]
    }   ,
    email:{
        type:String,
       
        // match:  [/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/, 'Please fill a valid email address']
    },
    password:{
        type:String,
        // required:[true,"Password is a required field"],
    },
    age:{
        type:Number
    }
    
},{timestamps:true})
Userschema.pre('save',async function (){
    const hashedpassword= await bcrypt.hash(this.password,10);
    this.password=hashedpassword
});
const User= mongoose.model("User",Userschema)
module.exports=User