const express = require("express");
const { PORT } = require("./config/serverConfig.js");
const connectDb = require("./config/dbConfig.js");
const userRouter = require("./routes/userRoutes.js");
const scrapRouter = require("./routes/scraproutes.js");
const { MongoClient } = require("mongodb");

const cookieParser = require("cookie-parser");
const authrouter = require("./routes/authRoutes.js");
const isLoggedIn = require("./validation/authValidator.js");
const productRouter = require("./routes/productRoutes.js");
const cors = require("cors");
const serverconfig = require("./config/serverConfig.js");

const app = express();

const uri = "mongodb+srv://kanikadebnath8759:uM6vVxAven34GOU6@cluster0.wopok.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";

const corsOptions = {
  origin: serverconfig.FRONTEND_URL,
  methods: "GET,HEAD,PUT,PATCH,POST,DELETE",
  credentials: true,
};

app.use(cors(corsOptions));
app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Initialize MongoDB client
const client = new MongoClient(uri);

app.listen(serverconfig.PORT, async () => {
  await connectDb();
  console.log(`Successfully connected to port ${serverconfig.PORT}`);
});

app.get("/ping", isLoggedIn, async (req, res) => {
  console.log(req.cookies);
  return res.json({ message: "pong" });
});

app.use("/auth", authrouter);
app.use("/users", userRouter);
app.use("/wastepicker", productRouter);
app.use("/buyScrap", scrapRouter);
app.use("/api/startups", require("./routes/startUpRoutes.js"));
app.use("/api/upload", require("./routes/uploadRoutes.js"));

app.post("/filter-ads", async (req, res) => {
  console.log("Request query:", req.query);
  console.log("Received body:", req.body);
  const { location, adType } = req.body;

  try {
    const database = client.db("test");
    const collection = database.collection("advertisement");

    const ads = await collection.find().toArray();
    console.log("All Ads:", ads);

    let filteredAds = [];

    // Traverse each document and filter based on location and/or adType
    ads.forEach(doc => {
      if (doc.Sheet1) {
        const matchedAds = doc.Sheet1.filter(ad => {
          // Check both conditions if both location and adType are provided
          if (location && adType) {
            return (
              ad.location.toLowerCase() === location.toLowerCase() &&
              ad.adType.toLowerCase() === adType.toLowerCase()
            );
          }
          // Check only location if adType is not provided
          else if (location) {
            return ad.location.toLowerCase() === location.toLowerCase();
          }
          // Check only adType if location is not provided
          else if (adType) {
            return ad.adType.toLowerCase() === adType.toLowerCase();
          }
          // If neither location nor adType is provided, return everything
          return true;
        });

        filteredAds = filteredAds.concat(matchedAds);
      }
    });

    console.log("Filtered Ads:", filteredAds);
    res.json({ message: "Filtered ads retrieved successfully", data: filteredAds });
  } catch (error) {
    console.error("Error fetching ads:", error);
    res.status(500).json({ message: "Server error", error });
  }
});


app.post("/filter-startups", async (req, res) => {
  console.log("Request query:", req.query);
  console.log("Received body:", req.body);
  const { startuptype } = req.body;

  try {
    const database = client.db("test");
    const collection = database.collection("startups");

    const startups = await collection.find().toArray();
    console.log("All Ads:", startups);
    

    let filteredStartups = [];

    // Traverse each document and filter based on location and/or adType
    startups.forEach(doc => {
      if (doc.Sheet1) {
        const matchedAds = doc.Sheet1.filter(ad => {
          // Check both conditions if both location and adType are provided
          if (startuptype) {
            return (
              ad.Type.toLowerCase() === startuptype.toLowerCase() 
            );
          }
          // else         
          // return true;
        });

        filteredStartups = filteredStartups.concat(matchedAds);
      }
    });

    console.log("Filtered Ads:", filteredStartups);
    res.json({ message: "Filtered ads retrieved successfully", data: filteredStartups });
  } catch (error) {
    console.error("Error fetching ads:", error);
    res.status(500).json({ message: "Server error", error });
  }
});

