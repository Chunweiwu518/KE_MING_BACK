import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

// 定義擴展的 Input 屬性類型
interface ExtendedInputHTMLAttributes extends React.InputHTMLAttributes<HTMLInputElement> {
  webkitdirectory?: string;
  directory?: string;
}

// 文本格式化函數
const formatText = (text: string): string => {
  // 先進行基本的清理
  let formattedText = text
    // 移除控制字符 (使用 Unicode 範圍而不是 hex)
    .replace(/[\u0000-\u001F\u007F-\u009F]/g, '')
    // 處理可能的 Unicode 轉義序列
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
    // 移除多餘的反斜線
    .replace(/\\([^u])/g, '$1')
    // 保留換行符
    .replace(/\n/g, '<br/>')
    // 替換標準的分隔符為HTML換行和列表項
    .replace(/-\s?\*\*([^*]+)\*\*:\s?/g, '<li><strong>$1</strong>: ')
    .replace(/\*\*([^*]+)\*\*:/g, '<strong>$1</strong>:')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    // 處理冒號後面的內容
    .replace(/(\d+)x(\d+)/g, '$1×$2')
    // 確保適當的列表包裹
    .replace(/<li>/g, '<li class="mb-2 list-disc ml-4">')
    // 讓產品標題更明顯
    .replace(/(HK-\d+的產品資料如下：)/g, '<div class="text-lg font-medium my-2">$1</div>')

  // 檢查是否有列表項，如果有則添加ul標籤
  if (formattedText.includes('<li>')) {
    formattedText = formattedText.replace(/<li>(.+?)(?=<li>|$)/g, '<ul><li>$1</ul>')
    // 修復嵌套的ul標籤
    formattedText = formattedText.replace(/<\/ul><ul>/g, '')
  }

  return formattedText
}

// 根據文本內容返回適當的CSS類
const getMessageStyle = (content: string, role: 'user' | 'assistant'): string => {
  if (role === 'user') {
    return 'bg-purple-600 text-white'
  }
  
  // 如果是產品資訊，增加更好的排版樣式
  if (content.includes('產品資料如下') || content.includes('商品名稱')) {
    return 'bg-gray-100 text-gray-800 product-info'
  }
  
  return 'bg-gray-100 text-gray-800'
}

interface FileInfo {
  name: string;
  display_name?: string;
  size?: number;
  lastModified?: number;
  uploadTime?: string;
  webkitRelativePath?: string;
  type?: string;
  status?: 'uploading' | 'success' | 'error';
  errorMessage?: string;
}

interface Source {
  content: string
  metadata: {
    source: string
    page?: number
  }
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
}

interface ChatHistory {
  id: string
  title: string
  messages: Message[]
  createdAt: string
}

const API_URL = import.meta.env.VITE_API_URL

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [files, setFiles] = useState<FileInfo[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [chatHistories, setChatHistories] = useState<ChatHistory[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [vectorStoreStats, setVectorStoreStats] = useState({
    total_chunks: 0,
    unique_files: 0,
    files: [],
    is_empty: true
  })

  // 載入歷史對話
  useEffect(() => {
    fetchChatHistories()
    // 添加獲取已上傳檔案的調用
    fetchUploadedFiles()
  }, [])

  // 滾動到最新消息
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 獲取已上傳的檔案列表
  const fetchUploadedFiles = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/files`)
      setFiles(response.data)
      console.log('已獲取上傳檔案列表:', response.data.length)
    } catch (error) {
      console.error('獲取上傳檔案列表失敗:', error)
    }
  }

  const fetchChatHistories = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/history`)
      // 按創建時間排序，最新的在前面
      const sortedHistories = response.data.sort((a: ChatHistory, b: ChatHistory) => 
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
      setChatHistories(sortedHistories)
    } catch (error) {
      console.error('Failed to fetch chat histories:', error)
    }
  }

  const loadChatHistory = async (chatId: string) => {
    try {
      setIsLoading(true)
      // 如果點擊當前對話，不需要重新載入
      if (chatId === currentChatId) {
        return
      }

      const response = await axios.get(`${API_URL}/api/history/${chatId}`)
      setMessages(response.data.messages || [])
      setCurrentChatId(chatId)
      
      // 更新 UI 狀態
      const selectedChat = chatHistories.find(chat => chat.id === chatId)
      if (selectedChat) {
        // 更新當前對話的標題等信息
        document.title = `RAG - ${selectedChat.title}`
      }
    } catch (error) {
      console.error('載入對話歷史失敗:', error)
      setError('載入對話歷史失敗')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      role: 'user',
      content: input.trim()
    }

    try {
      setIsLoading(true)
      setError(null)

      // 添加用戶消息到對話
      setMessages(prev => [...prev, userMessage])
      
      // 清空輸入
      setInput('')

      // 創建助手的臨時消息
      const assistantMessage: Message = {
        role: 'assistant',
        content: ''
      }
      
      // 添加助手的臨時消息
      setMessages(prev => [...prev, assistantMessage])

      // 使用當前的消息歷史（包括新的用戶消息）
      const currentHistory = messages.concat(userMessage)

      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage.content,
          history: currentHistory.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      let tempResponse = ''
      let sources: Source[] = []
      const decoder = new TextDecoder()

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('無法獲取響應流')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n')

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue

          const data = line.slice(6)

          if (data.startsWith('[SOURCES]') && data.endsWith('[/SOURCES]')) {
            const sourcesData = data.slice(9, -10)
            sources = JSON.parse(sourcesData)
          } else if (data.startsWith('[ERROR]')) {
            const errorMsg = data.replace('[ERROR]', '').replace('[/ERROR]', '')
            setError(`聊天請求失敗: ${errorMsg}`)
            break
          } else if (data === '[DONE]') {
            // 更新最終的助手消息
            setMessages(prev => {
              const updatedMessages = [...prev]
              const lastAssistantIndex = updatedMessages.length - 1
              if (lastAssistantIndex >= 0) {
                updatedMessages[lastAssistantIndex] = {
                  role: 'assistant',
                  content: tempResponse,
                  sources
                }
              }
              return updatedMessages
            })
            break
          } else {
            tempResponse += data
            // 即時更新助手的回應
            setMessages(prev => {
              const updatedMessages = [...prev]
              const lastAssistantIndex = updatedMessages.length - 1
              if (lastAssistantIndex >= 0) {
                updatedMessages[lastAssistantIndex] = {
                  ...updatedMessages[lastAssistantIndex],
                  content: tempResponse
                }
              }
              return updatedMessages
            })
          }
        }
      }

      // 如果這是新對話，保存歷史
      if (!currentChatId) {
        await saveOrUpdateChatHistory(
          messages.concat({
            role: 'assistant',
            content: tempResponse,
            sources
          }),
          userMessage.content.slice(0, 20) + "..."
        )
      }

    } catch (error) {
      console.error('聊天請求失敗:', error)
      setError('聊天請求失敗，請稍後再試')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = e.target.files
    if (!uploadedFiles) return

    setIsLoading(true)
    setError('正在處理文件...')
    let uploadSuccess = false;

    for(let i = 0; i < uploadedFiles.length; i++) {
      const file = uploadedFiles[i]
      const uploadFormData = new FormData()
      uploadFormData.append('file', file)
      
      // 添加一個臨時文件項，狀態為上傳中
      const tempFileId = Date.now() + '_' + i; // 創建一個臨時ID
      const tempFile: FileInfo = { 
        name: tempFileId,
        display_name: file.name,
        size: file.size,
        status: 'uploading'
      };
      
      setFiles(prev => [...prev, tempFile]);
      
      try {
        // 直接調用 API 不保留 response 變數
        await axios.post(`${API_URL}/api/upload`, uploadFormData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        // 上傳成功，移除臨時文件
        setFiles(prev => prev.filter(f => f.name !== tempFileId));
        uploadSuccess = true;
      } catch (error) {
        console.error('文件上傳失敗:', error);
        
        // 更新文件狀態為錯誤
        setFiles(prev => prev.map(f => {
          if (f.name === tempFileId) {
            return {
              ...f,
              status: 'error',
              errorMessage: '上傳失敗'
            };
          }
          return f;
        }));
        
        setError(`文件 ${file.name} 上傳失敗`);
      }
    }
    
    // 如果至少有一個文件上傳成功，則重新獲取文件列表
    if (uploadSuccess) {
      await fetchUploadedFiles();
    }
    
    setIsLoading(false);
    
    // 如果沒有錯誤提示，清除錯誤狀態
    if (!files.some(f => f.status === 'error')) {
      setError(null);
    }
  }

  const removeFile = async (index: number) => {
    const fileToRemove = files[index]
    try {
      await axios.delete(`${API_URL}/api/files/${fileToRemove.name}`)
      setFiles(prev => prev.filter((_, i) => i !== index))
      // 手動刷新知識庫統計
      await loadVectorStoreStats()
    } catch (error) {
      console.error('Delete error:', error)
      if (axios.isAxiosError(error)) {
        setError(`刪除檔案失敗: ${error.response?.data?.detail || error.message}`)
      } else {
        setError('刪除檔案時發生未知錯誤')
      }
    }
  }

  // 新增：刪除對話歷史
  const deleteHistory = async (chatId: string) => {
    try {
      await axios.delete(`${API_URL}/api/history/${chatId}`)
      await fetchChatHistories()
      if (currentChatId === chatId) {
        setMessages([])
        setCurrentChatId(null)
      }
    } catch (error) {
      console.error('Delete history error:', error)
      setError('刪除對話歷史失敗')
    }
  }

  // 修改 clearVectorStore 函數確保能徹底清空
  const clearVectorStore = async () => {
    if (!confirm('確定要清空知識庫嗎？此操作將刪除所有已學習的知識，且無法恢復。')) {
      return
    }
    
    setIsLoading(true)
    setError('正在清空知識庫...')
    
    try {
      await axios.delete(`${API_URL}/api/vector-store/clear`)
      // 清空後重新獲取檔案列表
      await fetchUploadedFiles()
      setError(null)
      await loadVectorStoreStats()
    } catch (error) {
      console.error('Clear vector store error:', error)
      if (axios.isAxiosError(error)) {
        setError(`清空知識庫失敗: ${error.response?.data?.detail || error.message}`)
      } else {
        setError('清空知識庫時發生未知錯誤')
      }
    } finally {
      setIsLoading(false)
    }
  }

  // 處理資料夾上傳
  const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsLoading(true)
    setError('正在處理資料夾中的文件...')

    let uploadedCount = 0
    let failedCount = 0
    let uploadSuccess = false

    // 處理所有文件
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      
      // 檢查副檔名
      const fileExt = file.name.toLowerCase().split('.').pop()
      if (!['txt', 'pdf', 'docx'].includes(fileExt || '')) continue
      
      // 添加一個臨時文件項，狀態為上傳中
      const tempFileId = `folder_${Date.now()}_${i}`; // 創建一個臨時ID
      const tempFile: FileInfo = { 
        name: tempFileId,
        display_name: file.name,
        size: file.size,
        status: 'uploading'
      };
      
      setFiles(prev => [...prev, tempFile]);
      
      try {
        const individualFormData = new FormData()
        individualFormData.append('file', file)
        
        // 直接調用 API 不保留 response 變數
        await axios.post(`${API_URL}/api/upload`, individualFormData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        // 上傳成功，移除臨時文件
        setFiles(prev => prev.filter(f => f.name !== tempFileId));
        uploadedCount++
        uploadSuccess = true
      } catch (error) {
        console.error(`上傳文件失敗: ${file.name}`, error)
        
        // 更新文件狀態為錯誤
        setFiles(prev => prev.map(f => {
          if (f.name === tempFileId) {
            return {
              ...f,
              status: 'error',
              errorMessage: '上傳失敗'
            };
          }
          return f;
        }));
        
        failedCount++
      }
    }

    // 如果至少有一個文件上傳成功，則重新獲取文件列表
    if (uploadSuccess) {
      await fetchUploadedFiles()
    }
    
    setIsLoading(false)
    if (failedCount > 0) {
      setError(`${uploadedCount} 個文件上傳成功，${failedCount} 個文件失敗`)
    } else if (uploadedCount === 0) {
      setError('沒有找到支持的文件類型 (PDF, TXT, DOCX)')
    } else {
      setError(null)
    }
    
    // 重置 input 控件，允許再次選擇相同文件
    if (folderInputRef.current) {
      folderInputRef.current.value = ''
    }
  }

  // 添加側邊欄收合切換函數
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  // 添加拖拽處理函數
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length === 0) return
    
    setIsLoading(true)
    setError('正在處理文件...')
    let uploadSuccess = false
    
    for(let i = 0; i < droppedFiles.length; i++) {
      const file = droppedFiles[i]
      
      // 檢查副檔名
      const fileExt = file.name.toLowerCase().split('.').pop()
      if (!['txt', 'pdf', 'docx'].includes(fileExt || '')) {
        setError(`不支持的文件類型: ${file.name}. 僅支持 PDF, TXT, DOCX`)
        continue
      }
      
      // 添加一個臨時文件項，狀態為上傳中
      const tempFileId = `drop_${Date.now()}_${i}`; // 創建一個臨時ID
      const tempFile: FileInfo = { 
        name: tempFileId,
        display_name: file.name,
        size: file.size,
        status: 'uploading'
      };
      
      setFiles(prev => [...prev, tempFile]);
      
      const dropFormData = new FormData()
      dropFormData.append('file', file)
      
      try {
        await axios.post(`${API_URL}/api/upload`, dropFormData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        // 上傳成功，移除臨時文件
        setFiles(prev => prev.filter(f => f.name !== tempFileId));
        uploadSuccess = true
      } catch (error) {
        console.error('文件上傳失敗:', error)
        
        // 更新文件狀態為錯誤
        setFiles(prev => prev.map(f => {
          if (f.name === tempFileId) {
            return {
              ...f,
              status: 'error',
              errorMessage: '上傳失敗'
            };
          }
          return f;
        }));
        
        setError(`文件 ${file.name} 上傳失敗`)
      }
    }
    
    // 如果至少有一個文件上傳成功，則重新獲取文件列表
    if (uploadSuccess) {
      await fetchUploadedFiles()
    }
    
    setIsLoading(false)
    // 如果沒有錯誤提示，清除錯誤狀態
    if (!files.some(f => f.status === 'error')) {
      setError(null)
    }
  }

  // 添加加載統計信息的函數
  const loadVectorStoreStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/vector-store/stats`)
      setVectorStoreStats(response.data)
    } catch (error) {
      console.error('獲取知識庫統計失敗:', error)
    }
  }

  // 在適當的時機加載統計信息
  useEffect(() => {
    loadVectorStoreStats()
  }, [files]) // 當文件列表變化時重新加載

  // 添加保存對話歷史的函數
  const saveOrUpdateChatHistory = async (messages: Message[], title: string) => {
    try {
      if (currentChatId) {
        // 更新現有對話
        await axios.put(`${API_URL}/api/history/${currentChatId}`, {
          messages,
          title
        })
      } else {
        // 創建新對話
        const response = await axios.post(`${API_URL}/api/history`, {
          messages,
          title
        })
        setCurrentChatId(response.data.id)
        // 更新對話歷史列表
        await fetchChatHistories()
      }
    } catch (error) {
      console.error('保存對話歷史失敗:', error)
      setError('保存對話歷史失敗')
    }
  }

  // 修改開始新對話的函數
  const startNewChat = () => {
    setMessages([])
    setCurrentChatId(null)
    setInput('')
    setError(null)
    document.title = 'RAG - 新對話'
  }

  // 添加在 App.tsx 中，用於渲染消息和來源
  const MessageContent: React.FC<{ message: Message }> = ({ message }) => {
    const [isSourcesVisible, setIsSourcesVisible] = useState(false);
    
    return (
      <div className="w-full">
        <div 
          className={`formatted-message ${getMessageStyle(message.content, message.role)}`}
          dangerouslySetInnerHTML={{ __html: formatText(message.content) }}
        />
        
        {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setIsSourcesVisible(!isSourcesVisible)}
              className="text-sm text-purple-600 hover:text-purple-800 flex items-center"
            >
              <span>{isSourcesVisible ? '隱藏來源' : '查看來源'}</span>
              <svg
                className={`ml-1 h-4 w-4 transform transition-transform ${
                  isSourcesVisible ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {isSourcesVisible && (
              <div className="mt-2 space-y-2">
                {message.sources.map((source, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 text-sm">
                    <div className="text-gray-600 mb-1">
                      來源文件：{source.metadata?.source || '未知來源'}
                      {source.metadata?.page && ` (第 ${source.metadata.page} 頁)`}
                    </div>
                    <div className="text-gray-800">{source.content}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* 側邊欄 */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} bg-white shadow-lg transition-all duration-300 overflow-hidden flex flex-col`}>
        <div className="p-4 border-b">
          <h2 className="text-lg font-semibold">RAG 聊天助手</h2>
        </div>
        
        {/* 上傳區域 */}
        <div className="p-4 border-b">
          <button
            onClick={() => document.getElementById('fileInput')?.click()}
            className="w-full bg-purple-600 text-white rounded-lg px-4 py-2 hover:bg-purple-700"
          >
            選擇檔案
          </button>
          <input
            id="fileInput"
            type="file"
            multiple
            className="hidden"
            onChange={handleFileUpload}
          />
        </div>

        {/* 對話歷史列表 */}
        <div className="flex-1 overflow-y-auto">
          {chatHistories.map((chat) => (
            <div
              key={chat.id}
              onClick={() => loadChatHistory(chat.id)}
              className={`p-4 cursor-pointer hover:bg-gray-50 ${
                currentChatId === chat.id ? 'bg-purple-50 border-l-4 border-purple-600' : ''
              }`}
            >
              <div className="text-sm font-medium">{chat.title}</div>
              <div className="text-xs text-gray-500">
                {new Date(chat.createdAt).toLocaleString()}
              </div>
            </div>
          ))}
        </div>

        {/* 新對話按鈕 */}
        <div className="p-4 border-t">
          <button
            onClick={startNewChat}
            className="w-full bg-gray-200 text-gray-700 rounded-lg px-4 py-2 hover:bg-gray-300"
          >
            開始新對話
          </button>
        </div>
      </div>

      {/* 主要聊天區域 */}
      <div className="flex-1 flex flex-col">
        {/* 頂部導航欄 */}
        <div className="bg-white shadow-sm p-4 flex items-center">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-gray-600 hover:text-gray-800"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="ml-4 text-lg font-semibold">RAG 知識庫問答</h1>
        </div>

        {/* 消息列表 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-lg px-4 py-2 ${
                  message.role === 'user' ? 'bg-purple-600 text-white' : 'bg-white'
                }`}
              >
                <MessageContent message={message} />
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* 錯誤提示 */}
        {error && (
          <div className="p-4 bg-red-100 text-red-700 text-sm">
            {error}
          </div>
        )}

        {/* 輸入區域 */}
        <div className="bg-white border-t p-4">
          <form onSubmit={handleSubmit} className="flex space-x-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="輸入問題..."
              className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-600"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className={`px-6 py-2 rounded-lg ${
                isLoading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-purple-600 hover:bg-purple-700'
              } text-white`}
            >
              {isLoading ? '處理中...' : '發送'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App