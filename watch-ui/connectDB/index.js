const mongoose = require('mongoose');

const connect = () => {
  mongoose
    .connect(
      `mongodb+srv://bm21btech11007:mZFEorxtcJXKIXg2@cluster0.aj4hxip.mongodb.net/?retryWrites=true&w=majority`,
      { useNewUrlParser: true, useUnifiedTopology: true }
    )
    .then(() => console.log('DB connection successful'));
};

export default connect;
