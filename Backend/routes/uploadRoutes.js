const express = require("express");
const multer = require("multer");
const csvParser = require("csv-parser");
const fs = require("fs");
const Startup = require("../schemas/startUp");

const router = express.Router();
const upload = multer({ dest: "uploads/" });

// Upload CSV and save data to MongoDB
router.post("/upload", upload.single("file"), (req, res) => {
  const filePath = req.file.path;
  const startups = [];

  fs.createReadStream(filePath)
    .pipe(csvParser())
    .on("data", (data) => startups.push(data))
    .on("end", async () => {
      try {
        await Startup.insertMany(startups);
        res.json({ message: "CSV Uploaded and Data Saved" });
      } catch (error) {
        res.status(500).send("Error saving data");
      }
      fs.unlinkSync(filePath);
    });
});

module.exports = router;
