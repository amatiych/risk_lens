import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppProvider } from "@/context/AppContext";
import { Sidebar } from "@/components/layout/Sidebar";
import { UploadPage } from "@/components/upload/UploadPage";
import { DashboardPage } from "@/components/dashboard/DashboardPage";
import { ChatPage } from "@/components/chat/ChatPage";

function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <div className="flex min-h-screen bg-background">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <Routes>
              <Route path="/" element={<UploadPage />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/chat" element={<ChatPage />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </AppProvider>
  );
}

export default App;
